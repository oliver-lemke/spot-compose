"""
This file contains the FrameTransformer and associated classes.
The FrameTransformer is important class to all movements, because it handles transformations between all kinds of
frames. Usually the FrameTreeSnapshot from Boston Dynamics handles frame transformations, however this class adds
some additional capabilities and is a bit more powerful.
Namely, the frame_tree_snapshot can't handle adding custom frames that well.
Additionally, the part of the frame tree we get back depends on the function we call. Typically, the different
snapshots are disjointed, leading to difficult transformations. This class serves to unify all transformations.
Finally, this class is persistent over the runtime of the program, allowing free access to all transformations at any
time. The main class here is the FrameTransformer.
"""

from __future__ import annotations

import numpy as np

from bosdyn.api import trajectory_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
)
from utils.coordinates import build_trajectory_point

GRAPH_SEED_FRAME_NAME = "seed_graph"
VISUAL_SEED_FRAME_NAME = "seed_visual"

from utils.singletons import (
    GraphNavClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
    _SingletonWrapper,
)

robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()
graph_nav_client = GraphNavClientSingleton()


class FrameTransformer:
    """
    Helper class designed to help with frame transformations of any kind.
    Persistent over runtime of program.
    """

    def __init__(self):
        self.robot_state_client = robot_state_client
        self.world_object_client = world_object_client
        self.graph_nav_client = graph_nav_client
        self.frames_tform_vision = {}

        frame_transformer = FrameTransformerSingleton()
        frame_transformer.set_instance(self)

    def transform(
        self,
        start_frame: str,
        end_frame: str,
        start_pose: math_helpers.SE2Pose | math_helpers.SE3Pose,
    ):
        """
        Basic transformation of a pose between two frames. Equivalent to end_tform_start
        :param start_frame: frame in which original pose is specified
        :param end_frame: frame to transform to
        :param start_pose: pose to transform (in start_frame)
        :return: pose in end_frame
        """
        end_tform_start = self.end_tform_start(start_frame, end_frame)
        if isinstance(start_pose, math_helpers.SE2Pose):
            end_tform_start = end_tform_start.get_closest_se2_transform()
        end_pose = end_tform_start * start_pose
        return end_pose

    def transform_wrench(
        self,
        start_frame: str,
        end_frame: str,
        start_wrench: trajectory_pb2.WrenchTrajectoryPoint,
    ):
        """
        Basic transformation of a pose between two frames. Equivalent to end_tform_start
        :param start_frame: frame in which original pose is specified
        :param end_frame: frame to transform to
        :param start_wrench: wrench to transform (in start_frame)
        :return: wrench in end_frame
        """
        end_tform_start = self.end_tform_start(start_frame, end_frame)

        force_in_start = start_wrench.wrench.force
        torque_in_start = start_wrench.wrench.torque
        t = start_wrench.time_since_reference.seconds
        force_in_end = end_tform_start.rotation.transform_point(
            force_in_start.x, force_in_start.y, force_in_start.z
        )
        torque_in_end = end_tform_start.rotation.transform_point(
            torque_in_start.x, torque_in_start.y, torque_in_start.z
        )

        return build_trajectory_point(
            t,
            force_in_end[0],
            force_in_end[1],
            force_in_end[2],
            torque_in_end[0],
            torque_in_end[1],
            torque_in_end[2],
        )

    def transform_matrix(self, start_frame: str, end_frame: str) -> np.ndarray:
        """
        Compute transformation matrix between two frames.
        :param start_frame: frame in which pose is originally specified
        :param end_frame: frame to transform to
        :return: end_tform_start
        """
        start_pose = math_helpers.SE3Pose(0, 0, 0, math_helpers.Quat())
        end_pose = self.transform(start_frame, end_frame, start_pose)
        return end_pose.to_matrix()

    def _get_frame_tform_body(self, frame_name: str) -> math_helpers.SE3Pose:
        """
        Returns the transformation from the given frame to the BODY_FRAME_NAME frame.
        Checks normal kinematics tree, SEED_FRAME tree, and world objects tree (if given).
        """
        # first  we check whether the frame is simply the SEED_FRAME, if so we return seed_tform_body
        if frame_name == GRAPH_SEED_FRAME_NAME:
            if not self.graph_nav_client:
                raise ValueError(
                    f"frame_name is {GRAPH_SEED_FRAME_NAME}, but graph_nav_client is not initialized!"
                )
            seed_tform_body = (
                self.graph_nav_client.get_localization_state().localization.seed_tform_body
            )
            return math_helpers.SE3Pose.from_proto(seed_tform_body)

        if frame_name in self.frames_tform_vision:
            odom_tform_body = self._get_frame_tform_body(ODOM_FRAME_NAME)
            return self.frames_tform_vision[frame_name] * odom_tform_body

        # then we want to check whether the frame is in the basic kinematics tree
        # if yes, we can simply return that
        robot_state = self.robot_state_client.get_robot_state()
        kinematics_transforms_snapshot = robot_state.kinematic_state.transforms_snapshot
        if frame_name in list(kinematics_transforms_snapshot.child_to_parent_edge_map):
            return get_a_tform_b(
                kinematics_transforms_snapshot, frame_name, BODY_FRAME_NAME
            )

        # last we check if the frame is in any of the world objects
        if self.world_object_client.is_instantiated():
            world_objects = self.world_object_client.list_world_objects().world_objects
            for world_object in world_objects:
                # check if one of the edges in the tree contains the wanted frame
                wo_transforms = world_object.transforms_snapshot
                edges = list(wo_transforms.child_to_parent_edge_map)
                if frame_name in edges:
                    return get_a_tform_b(wo_transforms, frame_name, BODY_FRAME_NAME)

        # if neither the base kinematics, nor the world objects tree contains the frame, it does not exist
        raise ValueError(
            f"Frame {frame_name} could not be found as transformation!\n {kinematics_transforms_snapshot=}"
        )

    def add_frame_tform_vision(
        self, name: str, tform_matrix: np.ndarray, overwrite: bool = False
    ):
        """
        Add a new frame to the transformer. Namely, frame_tform_odom.
        :param name: name of the new frame being added
        :param tform_matrix: frame_tform_odom
        :param overwrite: assuming a frame with name "name" already exists, overwrite it?
        """
        if name in self.frames_tform_vision and not overwrite:
            raise ValueError(f"Name {name} already in frame transformer!")
        self.frames_tform_vision[name] = math_helpers.SE3Pose.from_matrix(tform_matrix)

    def end_tform_start(self, start_frame: str, end_frame: str) -> math_helpers.SE3Pose:
        """
        Calculates transformation from start_frame to end_frame, checking kinematics, seed, and world objects trees.
        """
        # The problem is that all trees are generally disjointed. But all trees contain at the least a transformation
        # to the body. Therefore, we can use this fact to combine multiple trees via the body transformation.
        start_tform_body = self._get_frame_tform_body(start_frame)
        body_tform_end = self._get_frame_tform_body(end_frame).inverse()
        start_tform_end = start_tform_body * body_tform_end
        end_tform_start = start_tform_end.inverse()
        return end_tform_start

    def get_current_body_position_in_frame(
        self, end_frame: str
    ) -> math_helpers.SE2Pose:
        """
        Return current position of the robot in whatever frame was specified.
        :param end_frame: frame relative to which the pose should be returned
        """
        flat_body_center = math_helpers.SE2Pose(x=0, y=0, angle=0)
        return self.transform(
            start_frame=BODY_FRAME_NAME,
            end_frame=end_frame,
            start_pose=flat_body_center,
        )


########################################################################################################################
############################################ END OF FRAME_TRANSFORMER CLASS ############################################
########################################################################################################################


class FrameTransformerSingleton(_SingletonWrapper):
    """
    Singleton for FrameTransformer to allow for persistent storage and easy access.
    For more information on singleton see utils/singletons.py
    """

    _type_of_class = FrameTransformer
