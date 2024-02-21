"""
This class implement any utilities related to localization with the GraphNavService.
However, we don't currently use this file. That is because the Explorer version of Spot does not allow for simultaneous
use of both the GraphNavService and the depth camera in the gripper.
For that reason we use a custom localization, for that please see robot_utils/video.py.
This file has accordingly not been updated for some time. We only preserve it in case it could be useful in the future.
"""

from __future__ import annotations

import os.path

from bosdyn.api.geometry_pb2 import FrameTreeSnapshot, Quaternion, SE3Pose, Vec3
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.api.world_object_pb2 import WorldObject
from bosdyn.client import Robot
from bosdyn.client.frame_helpers import (
    VISION_FRAME_NAME,
    add_edge_to_tree,
    get_odom_tform_body,
)
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient, make_add_world_object_req
from bosdyn.util import now_timestamp
from robot_utils import frame_transformer as ft
from utils.recursive_config import Config


def set_initial_localization_fiducial(
    robot: Robot,
    robot_state_client: RobotStateClient,
    graph_nav_client: GraphNavClient,
    fiducial_id: int | None = None,
) -> None:
    """
    Trigger localization when near a fiducial.
    :param robot: Robot
    :param robot_state_client: RobotStateClient
    :param graph_nav_client: GraphNavClient
    :param fiducial_id: the specific fiducial ID to use for localization initialization,
    None uses nearest fiducial; it is recommended to use this field
    """
    if fiducial_id is None:
        robot.logger.warn("Fiducial ID for initial localization not set!")
    # adapted from graph_nav_command_line
    robot_state = robot_state_client.get_robot_state()
    current_odom_tform_body = get_odom_tform_body(
        robot_state.kinematic_state.transforms_snapshot
    ).to_proto()
    # Create an empty instance for initial localization since we are asking it to
    # localize based on the nearest fiducial.
    localization = nav_pb2.Localization()

    # whether to use a specific fiducial for init or just any nearest
    if fiducial_id is not None:
        fiducial_init = graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_SPECIFIC
        use_fiducial_id = fiducial_id
    else:
        fiducial_init = graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST
        use_fiducial_id = None

    graph_nav_client.set_localization(
        initial_guess_localization=localization,
        ko_tform_body=current_odom_tform_body,
        fiducial_init=fiducial_init,
        use_fiducial_id=use_fiducial_id,
        refine_with_visual_features=True,
    )


def upload_graph_and_snapshots(
    upload_file_path: bytes | str, robot: Robot, graph_nav_client: GraphNavClient
) -> None:
    """
    Upload the graph and snapshots to the robot.
    :param upload_file_path: The filepath of the autowalk (usually ends in .walk)
    :param robot: robot
    :param graph_nav_client: GraphNavClient
    """
    # TODO: Fails in here
    # adapted from graph_nav_command_line
    graph_path = os.path.join(str(upload_file_path), "graph")

    # parse the graph; start by reading the graph from the disk, then waypoint
    # snapshots, then edge snapshots
    with open(graph_path, "rb") as graph_file:
        # Load the graph from disk.
        data = graph_file.read()
        current_graph = map_pb2.Graph()
        current_graph.ParseFromString(data)
        robot.logger.info(
            f"Loaded graph has {len(current_graph.waypoints)} waypoints "
            f"and {len(current_graph.edges)} edges"
        )
        # robot.logger.info(current_graph.waypoints)

    current_waypoint_snapshots = {}
    for waypoint in current_graph.waypoints:
        # Load the waypoint snapshots from disk.
        waypoint_path = os.path.join(
            str(upload_file_path), "waypoint_snapshots", waypoint.snapshot_id
        )
        with open(waypoint_path, "rb") as snapshot_file:
            waypoint_snapshot = map_pb2.WaypointSnapshot()
            waypoint_snapshot.ParseFromString(snapshot_file.read())
            current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot

    current_edge_snapshots = {}
    for edge in current_graph.edges:
        if len(edge.snapshot_id) == 0:
            continue
        # Load the edge snapshots from disk.
        edge_path = os.path.join(
            str(upload_file_path), "edge_snapshots", edge.snapshot_id
        )
        with open(edge_path, "rb") as snapshot_file:
            edge_snapshot = map_pb2.EdgeSnapshot()
            edge_snapshot.ParseFromString(snapshot_file.read())
            current_edge_snapshots[edge_snapshot.id] = edge_snapshot

    # Upload the graph to the robot.
    true_if_empty = len(current_graph.anchoring.anchors) == 0
    # TODO: fails when uploading graph
    response = graph_nav_client.upload_graph(
        graph=current_graph, generate_new_anchoring=true_if_empty
    )
    # Upload the snapshots to the robot.
    for snapshot_id in response.unknown_waypoint_snapshot_ids:
        waypoint_snapshot = current_waypoint_snapshots[snapshot_id]
        graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
        # robot.logger.info(f"Uploaded {waypoint_snapshot.id}")
    for snapshot_id in response.unknown_edge_snapshot_ids:
        edge_snapshot = current_edge_snapshots[snapshot_id]
        graph_nav_client.upload_edge_snapshot(edge_snapshot)
        # robot.logger.info(f"Uploaded {edge_snapshot.id}")

    # The upload is complete. Check that the robot is localized to the graph,
    # and if it is not, prompt the user to localize the robot before attempting
    # any navigation commands.
    localization_state = graph_nav_client.get_localization_state()
    if not localization_state.localization.waypoint_id:
        # The robot is not localized to the newly uploaded graph.
        robot.logger.info(
            "Upload complete! The robot is currently not localized to the map; please"
            "localize the robot before attempting a navigation command."
        )


def add_graph_frame_to_transforms(
    world_object_client: WorldObjectClient,
    fiducial_id: int = 50,
    frame_name: str = "graph_frame",
) -> None:
    """
    Use the fiducial used for initial localization to root a new frame
    :param world_object_client: WorldObjectClient
    :param fiducial_id: The ID of the fiducial to base the frame from
    :param frame_name: name of the new frame
    """
    # adapted from https://dev.bostondynamics.com/docs/concepts/geometry_and_frames
    frame_tree_edges = {}  # Dictionary mapping child frame name to the
    # ParentEdge, which stores the parent frame name
    # and the SE3Pose parent_tform_child

    # Example of a special transform that we want to add to the robot.
    vision_tform_special_frame = SE3Pose(
        position=Vec3(x=1, y=1, z=1), rotation=Quaternion(w=1, x=0, y=0, z=0)
    )

    # Update the frame tree edge dictionary with the new transform using the helper
    # functions in frame_helpers.py.
    frame_tree_edges = add_edge_to_tree(
        frame_tree_edges, vision_tform_special_frame, VISION_FRAME_NAME, frame_name
    )

    # Pack the dictionary into a FrameTreeSnapshot proto message.
    snapshot = FrameTreeSnapshot(child_to_parent_edge_map=frame_tree_edges)

    # TODO: What exactly does this do (why is it used here)?
    # Create the world object containing the special frame.
    world_obj_special_frame = WorldObject(
        id=fiducial_id,
        name=frame_name,
        transforms_snapshot=snapshot,
        acquisition_time=now_timestamp(),
    )

    # Add the world object containing the special frame to Spot using a mutation request
    world_object_client.mutate_world_objects(
        mutation_req=make_add_world_object_req(world_obj_special_frame)
    )


def full_localize(
    config: Config, robot: Robot, robot_state_client: RobotStateClient
) -> (str, ft.FrameTransformer, GraphNavClient, WorldObjectClient):
    """ """
    raise DeprecationWarning(
        "Please do not use GraphNav localization! Use utils.video.localize_from_images() instead."
    )
    robot.logger.info("Localizing Robot.")
    # get client for graph nav
    graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
    graph_nav_client.clear_graph()

    # get path for low-res graph
    graph_path = os.path.join(
        str(config.get_subpath("autowalks")),
        f'{config["pre_scanned_graphs"]["low_res"]}.walk',
    )
    # upload graph data to local robot storage
    upload_graph_and_snapshots(graph_path, robot, graph_nav_client)
    # set nearest fiducial as localization base
    base_fiducial_id = config["pre_scanned_graphs"]["base_fiducial_id"]
    set_initial_localization_fiducial(
        robot, robot_state_client, graph_nav_client, fiducial_id=base_fiducial_id
    )

    # add additional transformation frame to robot tree
    world_obj_client = robot.ensure_client(WorldObjectClient.default_service_name)

    localization_state = graph_nav_client.get_localization_state()
    if not localization_state.localization.waypoint_id:
        robot.logger.info("Robot not localized")
        return
    else:
        robot.logger.info("Robot localized")

    transformer = ft.FrameTransformer(
        robot_state_client, world_obj_client, graph_nav_client
    )
    frame_name = ft.GRAPH_SEED_FRAME_NAME

    return frame_name, transformer, graph_nav_client, world_obj_client
