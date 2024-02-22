"""
This file contains very basic motions to be accessed by more complex movements.
"""

# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).
# adapted from frame_trajectory_command
from __future__ import annotations

import time

from bosdyn.api import geometry_pb2, robot_command_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.robot_command_pb2 import RobotCommand
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
from bosdyn.util import seconds_to_duration
from robot_utils.frame_transformer import FrameTransformerSingleton
from utils.coordinates import Pose2D, Pose3D, pose_distanced
from utils.singletons import (
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
)

frame_transformer = FrameTransformerSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()


def move_body(
    pose: Pose2D,
    frame_name: str,
    timeout: int = 10,
    stairs: bool = False,
) -> bool:
    """
    This is the most basic movement function. It simply tells the robot to go to a position in a specified frame.
    :param pose: the position to go to
    :param frame_name: frame relative to which the position is specified to
    :param timeout: seconds after which the movement is considered as failed
    :param stairs: hint whether to expect stairs
    :return: bool whether movement successful
    """
    pose = pose.as_pose()
    pose_t = frame_transformer.transform(frame_name, VISION_FRAME_NAME, pose)
    # build robot command
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=pose_t.x,
        goal_y=pose_t.y,
        goal_heading=pose_t.angle,
        frame_name=VISION_FRAME_NAME,
        # stair hint
        params=RobotCommandBuilder.mobility_params(stair_hint=stairs),
    )
    cmd_id = robot_command_client.robot_command(
        lease=None, command=robot_cmd, end_time_secs=time.time() + timeout
    )
    # Wait until the robot has reached the goal.
    while True:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = (
            feedback.feedback.synchronized_feedback.mobility_command_feedback
        )
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("Failed to reach the goal")
            return False
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (
            traj_feedback.status == traj_feedback.STATUS_AT_GOAL
            and traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED
        ):
            return True
        time.sleep(1)


def unstow_arm(body_assist: bool = False) -> None:
    """
    Put the arm in the "unstow" position.
    :param body_assist: whether to employ body_assist
    """
    if body_assist:
        # Define a stand command that we'll send if the IK service does not find a
        # solution
        body_control = spot_command_pb2.BodyControlParams(
            body_assist_for_manipulation=spot_command_pb2.BodyControlParams.BodyAssistForManipulation(
                enable_hip_height_assist=True, enable_body_yaw_assist=True
            )
        )
        body_assist_enabled_stand_command = RobotCommandBuilder.synchro_stand_command(
            params=spot_command_pb2.MobilityParams(body_control=body_control)
        )
        ready_command = RobotCommandBuilder.arm_ready_command(
            build_on_command=body_assist_enabled_stand_command
        )
    else:
        ready_command = RobotCommandBuilder.arm_ready_command()

    # Issue the command via the RobotCommandClient
    unstow_command_id = robot_command_client.robot_command(ready_command)

    block_until_arm_arrives(robot_command_client, unstow_command_id, 3.0)


def carry_arm(body_assist: bool = False) -> None:
    if body_assist:
        # Define a stand command that we'll send if the IK service does not find a
        # solution
        body_control = spot_command_pb2.BodyControlParams(
            body_assist_for_manipulation=spot_command_pb2.BodyControlParams.BodyAssistForManipulation(
                enable_hip_height_assist=True, enable_body_yaw_assist=True
            )
        )
        body_assist_enabled_stand_command = RobotCommandBuilder.synchro_stand_command(
            params=spot_command_pb2.MobilityParams(body_control=body_control)
        )
        carry_command = RobotCommandBuilder.arm_carry_command(
            build_on_command=body_assist_enabled_stand_command
        )
    else:
        carry_command = RobotCommandBuilder.arm_carry_command()

    # Issue the command via the RobotCommandClient
    carry_command_id = robot_command_client.robot_command(carry_command)

    block_until_arm_arrives(robot_command_client, carry_command_id, 3.0)


def stow_arm() -> None:
    """
    Put the arm in stowed position.
    """
    # Stow the arm
    # Build the stow command using RobotCommandBuilder
    stow = RobotCommandBuilder.arm_stow_command()

    # Issue the command via the RobotCommandClient
    stow_command_id = robot_command_client.robot_command(stow)

    # robot.logger.info("Stow command issued.")
    block_until_arm_arrives(robot_command_client, stow_command_id, 3.0)


def set_gripper(
    gripper_open: bool | float,
):
    """
    Set the gripper openness.
    :param gripper_open: can be float in [0.0, 1.0], False (=0.0) or True (=1.0)
    """
    if isinstance(gripper_open, bool):
        if gripper_open:
            fraction = 1.0
        else:
            fraction = 0.0
    else:
        fraction = gripper_open
    assert 0.0 <= fraction <= 1.0

    gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(fraction)

    # Send the request
    _ = robot_command_client.robot_command(gripper_command)
    # block_until_arm_arrives(robot_command_client, cmd_id)
    # TODO: this is a hack
    time.sleep(1)


def _normal_arm_move_command(
    pose: Pose3D, frame_name: str, root_frame: str, timeout: int
) -> RobotCommand:
    """
    Creates a normal arm movement command.
    """
    se3_pose = pose.as_pose()
    arm_command = RobotCommandBuilder.arm_pose_command(
        se3_pose.x,
        se3_pose.y,
        se3_pose.z,
        se3_pose.rot.w,
        se3_pose.rot.x,
        se3_pose.rot.y,
        se3_pose.rot.z,
        root_frame,
        timeout,
    )
    root_tform_task = frame_transformer.end_tform_start(frame_name, root_frame)
    arm_command.synchronized_command.arm_command.arm_cartesian_command.root_tform_task.CopyFrom(
        root_tform_task.to_proto()
    )
    return arm_command


def _impedance_arm_move_command(
    pose: Pose3D,
    stiffness_diag: list[int] | None,
    damping_diag: list[float] | None,
    forces: list[float] | None,
    frame_name: str,
    root_frame: str,
    duration: float,
) -> RobotCommand:
    """
    Creates an impedance arm command with the specified stiffness and damping.
    """

    robot_cmd = robot_command_pb2.RobotCommand()
    impedance_cmd = robot_cmd.synchronized_command.arm_command.arm_impedance_command

    # Set up our root frame, task frame
    impedance_cmd.root_frame_name = root_frame
    root_tform_task = frame_transformer.end_tform_start(frame_name, root_frame)
    impedance_cmd.root_tform_task.CopyFrom(root_tform_task.to_proto())

    # Set up forces at tool
    if forces:
        assert len(forces) == 6
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.force.x = forces[0]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.force.y = forces[1]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.force.z = forces[2]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.torque.x = forces[3]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.torque.y = forces[4]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.torque.z = forces[5]

    # Set up stiffness and damping matrices
    if stiffness_diag:
        assert len(stiffness_diag) == 6
        impedance_cmd.diagonal_stiffness_matrix.CopyFrom(
            geometry_pb2.Vector(values=stiffness_diag)
        )

    if damping_diag:
        assert len(damping_diag) == 6
        impedance_cmd.diagonal_damping_matrix.CopyFrom(
            geometry_pb2.Vector(values=damping_diag)
        )

    # Set up the target pose as the desired tool frame
    traj = impedance_cmd.task_tform_desired_tool
    pt1 = traj.points.add()
    pt1.time_since_reference.CopyFrom(seconds_to_duration(duration))
    pt1.pose.CopyFrom(pose.as_pose().to_proto())
    return robot_cmd


def move_arm(
    pose: Pose3D,
    frame_name: str,
    timeout: int = 5,
    gripper_open: bool | float | None = None,
    unstow: bool = False,
    stow: bool = False,
    body_assist: bool = False,
    keep_static_after_moving: bool = False,
    stiffness_diag: list[int] | None = None,
    damping_diag: list[float] | None = None,
    forces: list[float] | None = None,
) -> bool:
    """
    Moves the arm to a specified location relative to the body frame
    :param pose: pose to move arm to, specified both coordinates and rotation
    :param frame_name: frame relative to which pose is specified
    :param timeout: timeout for movement
    :param gripper_open: whether to open gripper with movement
    :param unstow: whether to unstow the arm before use
    :param stow: whether to stow the arm after use
    :param body_assist: whether to use body assist when grabbing
    :param keep_static_after_moving: if true, the robot will try to keep the arm at
    static position when moving, otherwise the arm will move with the robot
    :param stiffness_diag: diagonal stiffness matrix for impedance
    :param damping_diag: diagonal damping matrix for impedance
    :param forces: forces to apply to tool relative to the task frame
    """
    if unstow:
        unstow_arm(body_assist=body_assist)

    if keep_static_after_moving or body_assist:
        # For some reason body_assist needs the ODOM_FRAME_NAME
        if not keep_static_after_moving:
            robot.logger.warn("Arm kept static due to required body assist.")
        root_frame = VISION_FRAME_NAME
    else:
        root_frame = BODY_FRAME_NAME

    # start with stand command, which will serve as the basis of building
    body_control = spot_command_pb2.BodyControlParams(
        body_assist_for_manipulation=spot_command_pb2.BodyControlParams.BodyAssistForManipulation(
            enable_hip_height_assist=body_assist, enable_body_yaw_assist=body_assist
        )
    )
    stand_command = RobotCommandBuilder.synchro_stand_command(
        params=spot_command_pb2.MobilityParams(body_control=body_control)
    )

    # build the actual command
    if stiffness_diag is None and damping_diag is None:
        arm_command = _normal_arm_move_command(pose, frame_name, root_frame, timeout)
    else:
        arm_command = _impedance_arm_move_command(
            pose, stiffness_diag, damping_diag, forces, frame_name, root_frame, timeout
        )

    # Make the open gripper RobotCommand
    if gripper_open is not None:
        set_gripper(gripper_open)

    robot_cmd = RobotCommandBuilder.build_synchro_command(stand_command, arm_command)

    # Send the request
    cmd_id = robot_command_client.robot_command(robot_cmd)

    # Wait until the arm arrives at the goal.
    success = block_until_arm_arrives(robot_command_client, cmd_id)

    if stow:
        stow_arm()

    return success


def move_arm_distanced(
    pose: Pose3D,
    distance: float,
    frame_name: str,
    **kwargs,
) -> Pose3D:
    """
    Move the arm to a specified pose, but offset in the viewing direction.
    Imagine an axis in the viewing direction. That is the axis of movement along which "distance" specifies the exact
    position we move to.
    :param pose: theoretical pose (if distance = 0)
    :param distance: distance in m that the actual final pose is offset from the pose by
    :param frame_name: frame relative to which the position is specified
    :param kwargs: kwargs for the move_arm function
    """
    result_pos = pose_distanced(pose, distance)
    move_arm(pose=result_pos, frame_name=frame_name, body_assist=True, **kwargs)
    return result_pos


def roll_over():
    """
    Roll the robot into battery swap position.
    """
    command = RobotCommandBuilder.battery_change_pose_command()
    robot_command_client.robot_command(command)


def carry():
    """
    Put arm into carry position.
    """
    carry_cmd = RobotCommandBuilder.arm_carry_command()
    robot_command_client.robot_command(carry_cmd)
