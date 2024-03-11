"""
This code acts as the jump-off point for all later scripts.
It exists because there is a bunch of framework stuff we have to do before being able to control the robot
(including, but not limited to creating the sdk, syncing the robot, authenticating to the robot, verifying E-Stop, ...).
This code does all this framework code. For our custom code, you should create an object inheriting from
ControlFunction, in which the __call__() method defines the actual actions.
At the bottom of the script call take_control_with_function(f: ControlFunction, **kwargs).
"""

from __future__ import annotations

import time
import typing

from bosdyn import client as bosdyn_client
from bosdyn.api import estop_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import Sdk
from bosdyn.client import util as bosdyn_util
from bosdyn.client.estop import EstopClient
from bosdyn.client.robot_command import (
    RobotCommandClient,
    blocking_selfright,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient
from robot_utils.basic_movements import move_body
from robot_utils.frame_transformer import FrameTransformerSingleton
from utils import environment
from utils.coordinates import Pose3D
from utils.logger import LoggerSingleton, TimedFileLogger
from utils.recursive_config import Config
from utils.singletons import (
    GraphNavClientSingleton,
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
    reset_singletons,
)

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()
logger = LoggerSingleton()

ALL_SINGLETONS = (
    frame_transformer,
    graph_nav_client,
    image_client,
    robot_command_client,
    robot,
    robot_state_client,
    world_object_client,
)


class ControlFunction(typing.Protocol):
    """
    This class defines all control functions. It gets as input all that you need for controlling the robot.
    :return: FrameTransformer, FrameName used for returning to origin
    """

    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        pass


def take_control_with_function(
    config: Config,
    function: ControlFunction,
    *args,
    stand: bool = True,
    power_off: bool = True,
    body_assist: bool = False,
    return_to_start: bool = True,
    **kwargs,
):
    """
    Code wrapping all ControlFunctions, handles all kinds of framework (see description at beginning of file).
    :param config: config file for the robot
    :param function: ControlFunction specifying the actual actions
    :param args: other args for ControlFunction
    :param stand: whether to stand (and self-right if applicable) in beginning of script
    :param power_off: whether to power off after successful execution of actions
    :param body_assist: TODO: might not be useful at all
    :param return_to_start: whether to return to start at the end of execution
    :param kwargs: other keyword-args for ControlFunction
    """

    global logger
    logger.set_instance(TimedFileLogger(config))
    logger.log("Robot started")

    # Setup adapted from github.com/boston-dynamics/spot-sdk/blob/master/python/examples/hello_spot/hello_spot.py
    spot_env_config = environment.get_environment_config(config, ["spot"])
    robot_config = config["robot_parameters"]
    sdk = bosdyn_client.create_standard_sdk("understanding-spot")

    # setup logging
    bosdyn_util.setup_logging(robot_config["verbose"])

    # setup robot
    global robot
    robot.set_instance(sdk.create_robot(spot_env_config["wifi_default_address"]))
    environment.set_robot_password(config)
    bosdyn_util.authenticate(robot)

    # Establish time sync with the robot. This kicks off a background thread to establish time sync.
    # Time sync is required to issue commands to the robot. After starting time sync thread, block
    # until sync is established.
    robot.time_sync.wait_for_sync()

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    verify_estop()

    # The robot state client will allow us to get the robot's state information, and construct
    # a command using frame information published by the robot.
    global robot_state_client
    robot_state_client.set_instance(
        robot.ensure_client(RobotStateClient.default_service_name)
    )
    robot_state = robot_state_client.get_robot_state()
    # robot.logger.info(str(robot_state))

    # Only one client at a time can operate a robot. Clients acquire a lease to
    # indicate that they want to control a robot. Acquiring may fail if another
    # client is currently controlling the robot. When the client is done
    # controlling the robot, it should return the lease so other clients can
    # control it. The LeaseKeepAlive object takes care of acquiring and returning
    # the lease for us.
    lease_client = robot.ensure_client(
        bosdyn_client.lease.LeaseClient.default_service_name
    )

    ##################################################################################
    ################################## END OF SETUP ##################################
    ##################################################################################

    with bosdyn_client.lease.LeaseKeepAlive(
        lease_client, must_acquire=True, return_at_exit=True
    ):
        # This allows us to lease the robot, and in here we actually do the commands

        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        # robot.logger.info("Powering on robot... This may take several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        # robot.logger.info("Robot powered on.")

        battery_states = robot_state.battery_states[0]
        percentage = battery_states.charge_percentage.value
        estimated_time = battery_states.estimated_runtime.seconds / 60
        if percentage < 20.0:
            robot.logger.info(
                f"\033[91mCurrent battery percentage at {percentage}%.\033[0m"
            )
        else:
            robot.logger.info(f"Current battery percentage at {percentage}%.")
        robot.logger.info(f"Estimated time left {estimated_time:.2f} min.")

        # get command client
        global robot_command_client
        robot_command_client.set_instance(
            robot.ensure_client(RobotCommandClient.default_service_name)
        )

        if stand:
            # Here, we want the robot so self-right, otherwise it cannot stand
            # robot.logger.info("Commanding robot to self-right.")
            blocking_selfright(robot_command_client)
            # robot.logger.info("Self-righted")

            # Tell the robot to stand up. The command service is used to issue commands to a robot.
            # The set of valid commands for a robot depends on hardware configuration. See
            # RobotCommandBuilder for more detailed examples on command building. The robot
            # command service requires timesync between the robot and the client.
            # robot.logger.info("Commanding robot to stand...")

            if body_assist:
                body_control = spot_command_pb2.BodyControlParams(
                    body_assist_for_manipulation=spot_command_pb2.BodyControlParams.BodyAssistForManipulation(
                        enable_hip_height_assist=True, enable_body_yaw_assist=True
                    )
                )
                params = spot_command_pb2.MobilityParams(body_control=body_control)
            else:
                params = None
            blocking_stand(robot_command_client, timeout_sec=10, params=params)
            # robot.logger.info("Robot standing.")
            time.sleep(3)

        return_values = function(
            config,
            sdk,
            *args,
            **kwargs,
        )

        if return_to_start and return_values is not None:
            logger.log("Returning to start")
            frame_name = return_values
            return_pose = Pose3D((1.5, -0.1, 0))
            return_pose.set_rot_from_rpy((0, 0, 180), degrees=True)
            move_body(
                pose=return_pose.to_dimension(2),
                frame_name=frame_name,
            )

        logger.log("Fin.")
        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        if power_off:
            robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not robot.is_powered_on(), "Robot power off failed."
            robot.logger.info("Robot safely powered off.")

        reset_singletons(ALL_SINGLETONS)


def verify_estop():
    """Verify the robot is not estopped"""
    # https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/arm_joint_move/arm_joint_move.py

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = (
            "Robot is estopped. Please use an external E-Stop client, such as the "
            "estop SDK example, to configure E-Stop."
        )
        robot.logger.error(error_message)
        raise EStopError(error_message)


class EStopError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
