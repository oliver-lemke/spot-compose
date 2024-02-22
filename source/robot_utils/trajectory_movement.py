from __future__ import annotations

from bosdyn.api import (
    arm_command_pb2,
    robot_command_pb2,
    synchronized_command_pb2,
    trajectory_pb2,
)
from bosdyn.client.frame_helpers import VISION_FRAME_NAME
from robot_utils.basic_movements import stow_arm, unstow_arm
from robot_utils.frame_transformer import FrameTransformerSingleton
from utils.singletons import RobotCommandClientSingleton

robot_command_client = RobotCommandClientSingleton()
frame_transformer = FrameTransformerSingleton()


def move_arm_trajectory(
    traj_points: list[trajectory_pb2.WrenchTrajectoryPoint],
    frame_name: str,
    unstow: bool = False,
    stow: bool = False,
    body_assist: bool = False,
) -> None:
    """
    Moves along force trajectory
    :param traj_points: point along which the trajectory should move
    :param frame_name: frame relative to which trajectory is specified
    """
    if unstow:
        unstow_arm(body_assist=body_assist)

    intermediate_frame = VISION_FRAME_NAME
    end_traj_points = [
        frame_transformer.transform_wrench(frame_name, intermediate_frame, traj_point)
        for traj_point in traj_points
    ]

    trajectory = trajectory_pb2.WrenchTrajectory(points=end_traj_points)

    # Build the full request, putting all axes into force mode.
    arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
        root_frame_name=intermediate_frame,
        wrench_trajectory_in_task=trajectory,
        x_axis=arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_FORCE,
        y_axis=arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_FORCE,
        z_axis=arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_FORCE,
        rx_axis=arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_FORCE,
        ry_axis=arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_FORCE,
        rz_axis=arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_FORCE,
    )
    arm_command = arm_command_pb2.ArmCommand.Request(
        arm_cartesian_command=arm_cartesian_command
    )
    synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
        arm_command=arm_command
    )
    robot_command = robot_command_pb2.RobotCommand(
        synchronized_command=synchronized_command
    )
    robot_command_client.robot_command(robot_command)

    if stow:
        unstow_arm()
        stow_arm()
