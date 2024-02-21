"""
Conventions in Spot:
- x-coordinate means forward-backward translation, with positive being forward (meters)
- y-coordinate means left-right translation, with positive being left (meters)
- y-coordinate means up-down  translation, with positive being up (meters)
- yaw means rotation around z-axis, with positive being clockwise (radians)
Conventions for Gripper: (Quaternions)
"""

from __future__ import annotations

import abc
import math
from typing import Optional

import numpy as np

from bosdyn.api import geometry_pb2, trajectory_pb2
from bosdyn.api.geometry_pb2 import FrameTreeSnapshot
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.util import seconds_to_duration
from scipy.spatial.transform import Rotation


def _pose_from_dimension(dim):
    """
    Return class of pose from the dimension.
    """
    if dim == 2:
        return Pose2D
    elif dim == 3:
        return Pose3D


class Pose:
    """
    Parent class for the pose classes.
    In general, pose classes represent a spatial coordinate, as well as a rotation.
    This class is useful, because it allows us to seamlessly transform between different
    representations, such as
    (1) For coordinates: tuples, numpy arrays
    (2) For coordinates + rotation: transformation matrix, boston dynamics pose, etc.
    Additionally, we can set the rotation from the direction, rpy, matrix, etc.
    This makes working with different representations easier.
    """

    def __init__(self, dimension: int, coordinates: np.ndarray, rot_matrix: np.ndarray):
        self.dimension: int = dimension
        self.coordinates: np.ndarray = coordinates
        self.rot_matrix: np.ndarray = rot_matrix

    def as_tuple(self) -> tuple:
        return tuple(self.coordinates.tolist())

    def as_ndarray(self) -> np.ndarray:
        return self.coordinates

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(self.dimension + 1)
        matrix[: self.dimension, : self.dimension] = self.rot_matrix
        matrix[: self.dimension, self.dimension] = self.coordinates
        return matrix

    def transform(self, tform: np.ndarray, side: str = "left") -> None:
        self_mat = self.as_matrix()
        if side == "left":
            new_mat = tform @ self_mat
        elif side == "right":
            new_mat = self_mat @ tform
        else:
            raise ValueError(f"Unknown side {side}!")
        self.rot_matrix = new_mat[:3, :3]
        self.coordinates = new_mat[:3, 3]

    @abc.abstractmethod
    def as_pose(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_dimension(self, dimension: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def direction(self, normalized: bool = True) -> np.ndarray:
        raise NotImplementedError()

    def __str__(self):
        coords = (f"{coord:.2f}" for coord in self.coordinates)
        coords = f'({", ".join(coords)})'
        direction = (f"{x:.2f}" for x in self.direction())
        direction = f'({", ".join(direction)})'
        return f"Pose{self.dimension}D(coords={coords}, direction={direction})"

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        PoseClass = _pose_from_dimension(self.dimension)
        return PoseClass(self.coordinates.copy(), self.rot_matrix.copy())

    def copy(self):
        return self.__copy__()

    def __matmul__(self, other):
        assert isinstance(other, Pose), f"other is type {type(other)}!"
        max_dim = max(self.dimension, other.dimension)
        # cast to higher dimension
        if max_dim > self.dimension or max_dim > other.dimension:
            this = self.to_dimension(max_dim)
            other = other.to_dimension(max_dim)
        else:
            this = self

        rot_matrix = this.as_matrix() @ other.as_matrix()

        return _pose_from_dimension(max_dim).from_matrix(rot_matrix)


def _rot_matrix_from_angle(angle: float) -> np.ndarray:
    """
    Compute 2D rotation matrix from angle
    :return: 2D rotation matrix
    """
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def _angle_from_rot_matrix(rot_matrix: np.ndarray) -> float:
    """
    Compute angle from 2D rotation matrix
    :return: angle
    """
    return math.atan2(rot_matrix[1, 0], rot_matrix[0, 0])


def _rotation_from_direction(
    direction: (float, float, float),
    roll: float,
    invariant_direction: (float, float, float),
    degrees: bool,
    invert: bool = True,
) -> np.ndarray:
    """
    Determine the position from which to grasp (as well as the corresponding gripper
    rotation) from the direction from which to grasp
    :param direction: from which to grasp the object (center is (0, 0, 0))
    :param invariant_direction: where to start transformation (direction in which the
    transformation is pretty much invariant))
    :param roll: of gripper
    :param use_degrees: whether roll in degrees
    :return: start position of grasping motion, rotation matrix of gripper
    """
    direction = np.asarray(direction)
    invariant_direction = np.asarray(invariant_direction)

    start_vector = invariant_direction.reshape((1, 3))
    end_vector = direction.reshape((1, 3))
    if invert:
        pitch_yaw_rotation = Rotation.align_vectors(-end_vector, start_vector)[0]
    else:
        pitch_yaw_rotation = Rotation.align_vectors(end_vector, start_vector)[0]

    if degrees:
        roll = roll / 360 * math.tau

    roll_rotation = Rotation.from_euler("xyz", [roll, 0, 0])
    full_rotation = pitch_yaw_rotation * roll_rotation

    # Calculate the rotation between the two vectors
    return full_rotation.as_matrix()


class Pose2D(Pose):
    """
    This pose encodes (x, y) coordinates and an angle for rotation.
    """

    def __init__(
        self,
        coordinates: Optional[(tuple | np.ndarray)] = None,
        rot_matrix: Optional[(np.ndarray | float)] = None,
    ):
        self.dimension = 2
        coordinates = self.compute_coordinates(coordinates)
        rot_matrix = self.compute_rot_matrix(rot_matrix)
        super().__init__(self.dimension, coordinates, rot_matrix)

    def compute_coordinates(
        self, coordinates: Optional[(tuple | np.ndarray)]
    ) -> np.ndarray:
        """
        Computes the coordinates from different representations
        :param coordinates: the input representation, can be tuple or ndarray; shape (2,)
        :return: coordinates in ndarray representation; shape (2,)
        """
        if coordinates is None:
            coordinates = np.zeros((self.dimension,))
        elif isinstance(coordinates, tuple):
            coordinates = np.asarray(coordinates)
        elif isinstance(coordinates, np.ndarray):
            coordinates = coordinates.reshape((-1,))
        elif isinstance(coordinates, math_helpers.SE2Pose):
            coordinates = np.asarray((coordinates.x, coordinates.y))
        assert coordinates.shape == (self.dimension,)
        return coordinates

    def compute_rot_matrix(
        self, rot_matrix: Optional[(np.ndarray | float)]
    ) -> np.ndarray:
        """
        Computes the rotation matrix from different representations
        :param rot_matrix: the input representation, can be float or ndarray; shape (2, 2)
        :return: coordinates in ndarray representation; shape (2, 2)
        """
        if rot_matrix is None:
            rot_matrix = np.eye(self.dimension)
        else:
            if isinstance(rot_matrix, np.ndarray):
                rot_matrix = rot_matrix
            elif isinstance(rot_matrix, float):
                rot_matrix = _rot_matrix_from_angle(rot_matrix)
        assert rot_matrix.shape == (self.dimension, self.dimension)
        return rot_matrix

    def as_pose(self) -> math_helpers.SE2Pose:
        """
        Computes the boston dynamics pose.
        """
        x, y = self.coordinates.tolist()
        return math_helpers.SE2Pose(
            x=x, y=y, angle=_angle_from_rot_matrix(self.rot_matrix)
        )

    def to_dimension(self, dimension: int):
        """
        Transform from Pose2D to Pose2D or Pose3D.
        """
        if dimension == 2:
            return self
        elif dimension == 3:
            coordinates = np.zeros((3,))
            coordinates[:2] = self.coordinates[:2]
            rot_matrix = np.eye(3)
            rot_matrix[:2, :2] = self.rot_matrix
            return Pose3D(coordinates, rot_matrix)

    def direction(self, normalized: bool = True) -> np.ndarray:
        """
        Compute the direction in which the rotation "looks".
        """
        result = self.rot_matrix @ np.asarray([1, 0])
        if normalized:
            result = result / np.linalg.norm(result, 2)
        return result

    def set_rot_from_angle(self, angle: float, degrees: bool = False) -> None:
        """
        Set the rotation matrix from the rotation angle.
        """
        rot_matrix = Rotation.from_euler("z", angle, degrees=degrees).as_matrix()
        self.rot_matrix = rot_matrix[:2, :2]

    @staticmethod
    def from_bosdyn_pose(pose: math_helpers.SE2Pose) -> Pose2D:
        """
        Initialize Pose2D from the Boston Dynamics pose.
        """
        assert isinstance(pose, math_helpers.SE2Pose)
        coordinates = (pose.x, pose.y)
        rot_matrix = _rot_matrix_from_angle(pose.angle)
        return Pose2D(coordinates, rot_matrix)

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> Pose2D:
        """
        Initialize Pose2D from a (3, 3) transformation matrix
        """
        assert matrix.shape == (3, 3)
        return Pose2D(matrix[:2, 2], matrix[:2, :2])

    @staticmethod
    def from_transforms_snapshot(
        transforms_snapshot: FrameTreeSnapshot,
        relative_to_frame: str,
        frame_name_image_sensor: str,
    ) -> Pose2D:
        """
        Initialize Pose2D using the transformation from two frames in the frame_tree_snapshot.
        """
        print(f"{transforms_snapshot=}")
        print(f"{relative_to_frame=}")
        print(f"{frame_name_image_sensor=}")
        frame_tform_sensor = get_a_tform_b(
            transforms_snapshot, relative_to_frame, frame_name_image_sensor
        )
        return Pose2D.from_bosdyn_pose(frame_tform_sensor)


class Pose3D(Pose):
    def __init__(
        self,
        coordinates: Optional[(tuple | np.ndarray)] = None,
        rot_matrix: Optional[(np.ndarray | math_helpers.Quat)] = None,
    ):
        self.dimension = 3
        coordinates = self.compute_coordinates(coordinates)
        rot_matrix = self.compute_rot_matrix(rot_matrix)
        super().__init__(self.dimension, coordinates, rot_matrix)

    def compute_coordinates(
        self, coordinates: Optional[(tuple | np.ndarray)]
    ) -> np.ndarray:
        """
        Computes the coordinates from different representations
        :param coordinates: the input representation, can be tuple or ndarray; shape (3,)
        :return: coordinates in ndarray representation; shape (3,)
        """
        if coordinates is None:
            coordinates = np.zeros((self.dimension,))
        elif isinstance(coordinates, tuple):
            coordinates = np.asarray(coordinates)
        elif isinstance(coordinates, np.ndarray):
            coordinates = coordinates.reshape((-1,))
        assert coordinates.shape == (self.dimension,)
        return coordinates

    def compute_rot_matrix(
        self, rot_matrix: Optional[(np.ndarray | math_helpers.Quat)]
    ) -> np.ndarray:
        """
        Computes the rotation matrix from different representations
        :param rot_matrix: the input representation, BD Quat or ndarray representation; shape (3, 3)
        :return: coordinates in ndarray representation; shape (3, 3)
        """
        if rot_matrix is None:
            rot_matrix = np.eye(self.dimension)
        else:
            if isinstance(rot_matrix, np.ndarray):
                rot_matrix = rot_matrix
            elif isinstance(rot_matrix, math_helpers.Quat):
                rot_matrix = rot_matrix.to_matrix()
        assert rot_matrix.shape == (self.dimension, self.dimension)
        return rot_matrix

    def as_pose(self) -> math_helpers.SE3Pose:
        """
        Compute BD SE3Pose.
        """
        x, y, z = self.coordinates.tolist()
        return math_helpers.SE3Pose(
            x=x, y=y, z=z, rot=math_helpers.Quat.from_matrix(self.rot_matrix)
        )

    def to_dimension(self, dimension: int):
        """
        Convert from Pose3D to Pose2D or Pose3D.
        """
        if dimension == 2:
            coordinates = self.coordinates[:2]
            _, _, yaw = Rotation.from_matrix(self.rot_matrix).as_euler("xyz")
            rot_matrix = _rot_matrix_from_angle(yaw)
            return Pose2D(coordinates, rot_matrix)
        elif dimension == 3:
            return self

    def set_rot_from_rpy(
        self, rpy: (float, float, float), degrees: bool = False
    ) -> None:
        """
        Set rotation from roll, pitch, yaw
        :param rpy: (roll, pitch, yaw)
        :param degrees: whether roll, pitch, yaw is given in degrees
        """
        self.rot_matrix = Rotation.from_euler("xyz", rpy, degrees=degrees).as_matrix()

    def set_rot_from_direction(
        self,
        direction: (float, float, float),
        roll: float = 0,
        invariant_direction: (float, float, float) = (1, 0, 0),
        degrees: bool = False,
    ) -> None:
        """
        Set rotation from the direction in which we should look.
        :param direction: in which to look
        :param roll: around the direction
        :param invariant_direction: vector around which roll rotates initially, default (1, 0, 0)
        :param degrees: whether roll is given in degrees
        """
        self.rot_matrix = _rotation_from_direction(
            direction, roll, invariant_direction, degrees, invert=False
        )

    def set_from_scipy_rotation(self, rotation: Rotation) -> None:
        """
        Set rotation from scipy.spatial.transform.Rotation.
        """
        rot_matrix = rotation.as_matrix()
        self.rot_matrix = rot_matrix

    def direction(self, normalized: bool = True) -> np.ndarray:
        """
        Compute direction in which we "look"
        :param normalized: whether to normalize to unit vector
        :return: ndarray; shape (3,)
        """
        result = self.rot_matrix @ np.asarray([1, 0, 0])
        if normalized:
            result = result / np.linalg.norm(result, 2)
        return result

    def inverse(self, inplace: bool = False) -> Pose3D:
        """
        Invert the transformation encoded by Pose3D.
        :param inplace: whether to change current Pose3D
        :return: inverted Pose3D
        """
        matrix = self.as_matrix()
        matrix_inv = np.linalg.inv(matrix)
        if inplace:
            self.rot_matrix = matrix_inv[:3, :3]
            self.coordinates = matrix_inv[:3, 3]
        return self.from_matrix(matrix_inv)

    @staticmethod
    def from_bosdyn_pose(pose: math_helpers.SE3Pose) -> Pose3D:
        """
        Initialize Pose3D from BD SE3Pose.
        """
        assert isinstance(pose, math_helpers.SE3Pose), f"{type(pose)=}"
        coordinates = (pose.x, pose.y, pose.z)
        rot_matrix = pose.rot.to_matrix()
        return Pose3D(coordinates, rot_matrix)

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> Pose3D:
        """
        Initialize Pose3D from (4, 4) transformation matrix.
        """
        assert matrix.shape == (4, 4)
        return Pose3D(matrix[:3, 3], matrix[:3, :3])

    @staticmethod
    def from_transforms_snapshot(
        transforms_snapshot: FrameTreeSnapshot,
        relative_to_frame: str,
        frame_name_image_sensor: str,
    ) -> Pose3D:
        """
        Initialize Pose3D using the transformation from two frames in the frame_tree_snapshot.
        """
        frame_tform_sensor = get_a_tform_b(
            transforms_snapshot, relative_to_frame, frame_name_image_sensor
        )
        return Pose3D.from_bosdyn_pose(frame_tform_sensor)

    @staticmethod
    def from_scipy_rotation(rotation: Rotation) -> Pose3D:
        """
        Initialize Pose3D from scipy.spatial.transform.Rotation.
        """
        pose = Pose3D()
        pose.set_from_scipy_rotation(rotation)
        return pose


def from_a_to_b_distanced(
    start_pose: Pose2D, end_pose: Pose2D, distance: float
) -> Pose2D:
    """
    Starting from start_pose, calculate a pose that is distance units away from end_pose
    :param start_pose:
    :param end_pose:
    :param distance:
    :return: Pose2D of the pose that is distanced from the end_pose
    """
    start = start_pose.as_ndarray()
    end = end_pose.as_ndarray()

    # calculate yaw
    path = end - start
    yaw = math.atan2(path[1], path[0])

    # distanced position
    path_norm = np.linalg.norm(path)
    wanted_norm = path_norm - distance
    rescaled_path = path / path_norm * wanted_norm
    destination = start + rescaled_path

    return Pose2D(destination, yaw)


def pose_distanced(
    pose: Pose3D,
    distance: float,
    negate: bool = True,
) -> Pose3D:
    """
    From the actual grasp position, determine a distanced one (coming from the same direction)
    :param pose: grasp pose
    :param distance: from which to start the motion
    :param negate: negate direction (for gripper)
    :return: start position of gripper
    """
    m = -1 if negate else 1
    # simply offset by direction in which we look
    grasp_start_point_without_offset = m * pose.direction(normalized=True) * distance
    grasp_start_point = grasp_start_point_without_offset + pose.as_ndarray()
    return Pose3D(grasp_start_point, pose.rot_matrix)


def _cartesian_to_polar(vector: np.ndarray) -> np.ndarray:
    """
    Convert from cartesian to polar coordinates.
    """
    r = np.linalg.norm(vector)

    # Calculate the inclination angle (θ) using arccos
    theta = np.arccos(vector[2] / r)

    # Calculate the azimuthal angle (φ) using arctan2
    phi = np.arctan2(vector[1], vector[0])
    return np.asarray([r, theta, phi])


def _polar_to_cartesian(vector: np.ndarray) -> np.ndarray:
    """
    Convert from cartesian to polar coordinates.
    """
    r = vector[..., 0]
    theta = vector[..., 1]
    phi = vector[..., 2]
    # [r, theta, phi] = vector.tolist()
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def angle_views_from_target(
    start_pose: Pose3D,
    target_pose: Pose3D,
    nr_captures: int,
    increments: float,
    degrees: bool = False,
) -> list[Pose3D]:
    """
    Given a start_pose and a target_pose, return 2 * nr_captures poses horizontally around the start that all look at the
    target. Imagine a sphere around the target with radius of the distance between the target and the start. Now, on
    that sphere, imagine a horizontal line around the start. The poses will lie on that line.
    Assumes the start_pose looks directly at the target_pose.
    :param start_pose: current coordinates of the gripper
    :param target_pose: likely coordinates of the target object
    :param nr_captures: how many additional angles to capture left *and* right
    :param increments: offset (in radians / degrees) of vector (2), (3) from the main vector
    :param degrees: whether offset is given in degrees (True) or radians (False)
    """
    target_coordinates = target_pose.as_ndarray()
    start_rot_matrix = start_pose.rot_matrix
    distance = np.linalg.norm(target_coordinates - start_pose.as_ndarray())

    if degrees:
        increments = np.radians(increments)

    """
    Image the start_pose and translate it to the target_pose.
    In order to compute the poses, we have to rotate that translated pose by some yaw.
    Once this translated pose is rotated, we can simply translate it back in the opposite of the viewing direction
    by the original radius.
    """
    most_negative_offset = -(nr_captures * increments)
    poses = []
    for i in range(2 * nr_captures + 1):
        rot_offset = most_negative_offset + i * increments
        yaw_matrix = Rotation.from_euler("xyz", (0, 0, rot_offset)).as_matrix()
        rot_matrix = start_rot_matrix @ yaw_matrix
        new_pose = Pose3D(target_coordinates, rot_matrix)
        poses.append(pose_distanced(new_pose, distance, negate=True))

    return poses


def spherical_angle_views_from_target(
    start_pose: Pose3D,
    target_pose: Pose3D,
    nr_captures: int,
    offset: float,
    degrees: bool = False,
    include_start_pose: bool = True,
) -> list[Pose3D]:
    """
    Given a start_pose and a target_pose, return nr_captures poses spherically around the start that all look at the
    target. Imagine a sphere around the target with radius of the distance between the target and the start. Now, on
    that sphere, imagine a circle around the start. The poses will lie on that circle.
    Assumes the start_pose is "looking at" the target_pose.
    :param start_pose: Pose3D describing the start position
    :param target_pose: Pose3D describing the target position (center of sphere)
    :param nr_captures: number of poses calculated on the circle (equidistant on it)
    :param offset: offset from target to circle as an angle seen from the center
    :param degrees: whether offset is given in degrees
    :param include_start_pose: whether to include the start pose as a pose in the return
    :return: list of calculated poses
    """
    # TODO: Make the start_pose not have to look at the target_pose
    target_coordinates = target_pose.as_ndarray()
    start_rot_matrix = start_pose.rot_matrix
    distance = np.linalg.norm(target_coordinates - start_pose.as_ndarray())

    if degrees:
        offset = np.radians(offset)

    """
    Image the start_pose and translate it to the target_pose.
    In order to compute the poses, we have to rotate that translated pose by some yaw and pitch.
    In order for them to lie on a circle the equation (pitch**2 + yaw**2) = 1 must hold.
    So the pitch is basically the offset "horizontally" and the yaw "vertically".
    Once this translated pose is rotated, we can simply translate it back in the opposite of the viewing direction
    by the original radius.
    """
    # thetas describes the circle coordinate in circle angles
    thetas = np.linspace(0, 2 * np.pi, nr_captures + 1)[:-1]
    rolls = np.zeros(thetas.shape)
    pitchs = np.sin(thetas) * offset
    yaws = -np.cos(thetas) * offset
    rpys = np.stack((rolls, pitchs, yaws), axis=1)

    poses = [start_pose] if include_start_pose else []
    for rpy in rpys:
        pitch_yaw_matrix = Rotation.from_euler("xyz", rpy).as_matrix()
        rot_matrix = start_rot_matrix @ pitch_yaw_matrix
        # rotate translated pose
        new_pose = Pose3D(target_coordinates, rot_matrix)
        # translate pose in reverted viewing direction
        poses.append(pose_distanced(new_pose, distance, negate=True))

    return poses


def get_uniform_sphere_directions(
    resolution: int = 4, return_cartesian: bool = True
) -> np.ndarray:
    """
    Return directions uniformly distributed on a sphere
    :param resolution: number of directions for one circle slice along the equator
    :param return_cartesian: whether to return directions as cartesian coordinates
    :return: (n/2 + 1, n, 3) vector with vector coordinates of directions (on unit
    sphere), where n := resolution.
    Axis 0 grows along theta, axis 1 grows along phi. (see sphere coordinates).
    Vector format is (r, theta, phi).
    """
    assert resolution % 2 == 0, "Resolution should be even!"
    resolution = resolution // 2
    r = np.ones((resolution, resolution + 1))
    theta = -np.linspace(0, math.pi, resolution + 1)
    phi = np.linspace(0, math.pi, resolution + 1)[:-1]
    thetas, phis = np.meshgrid(theta, phi)
    directions = np.stack((r, thetas, phis), axis=-1)
    directions = np.transpose(directions, (1, 0, 2))

    opposite_directions = np.copy(directions)
    opposite_directions[..., 2] = opposite_directions[..., 2] + math.pi

    full_directions = np.concatenate([directions, opposite_directions], axis=1)

    if return_cartesian:
        full_directions = _polar_to_cartesian(  # pylint:disable=redefined-variable-type
            full_directions
        )
    return full_directions


def remove_duplicate_rows(arr, tolerance=None):
    """
    Remove duplicate rows from a 2D NumPy array.

    This function is designed to work with arrays where each row represents a 3D point,
    hence it is expected to have three columns (for x, y, z coordinates). It offers an
    option to specify a 'tolerance' level for approximate deduplication.

    :param arr: a 2D NumPy array where each row represents a 3D point.
    :param tolerance: a value defining the precision for considering two points as duplicates. If provided, the array
    elements are rounded to the nearest multiple of this value before identifying duplicates. This is useful for
    grouping points that are very close to each other but not exactly identical, typically due to numerical precision
    issues.

    :return: a 2D NumPy array similar to 'arr' but with duplicate rows removed. If 'tolerance' is specified, duplicates
    are identified based on the rounded values.
    """
    if tolerance:
        arr = np.round(arr / tolerance)

    arr_view = arr.view(dtype=[("f0", arr.dtype), ("f1", arr.dtype), ("f2", arr.dtype)])
    _, unique_idx = np.unique(arr_view, return_index=True)

    return arr[unique_idx]


def get_circle_points(
    resolution: int,
    nr_circles: int,
    start_radius: float = 0.0,
    end_radius: float = 1.0,
    return_cartesian: bool = True,
) -> np.ndarray:
    """
    Return points in the horizontal plane around a center. Distributed in circles.
    :param resolution: how many points per circle
    :param nr_circles: how many circles
    :param start_radius: radius for smallest circles
    :param end_radius: radius for largest circles
    :param return_cartesian: whether returned coordinates are cartesian
    """
    radius = np.linspace(start_radius, end_radius, nr_circles)
    thetas = np.ones((resolution, nr_circles)) * np.pi / 2
    phi = np.linspace(0, math.tau, resolution + 1)[:-1]
    radii, phis = np.meshgrid(radius, phi)
    vectors = np.stack((radii, thetas, phis), axis=-1)
    vectors = np.transpose(vectors, (1, 0, 2))

    if return_cartesian:
        vectors = _polar_to_cartesian(vectors)
    return vectors


def build_trajectory_point(
    seconds: float,
    force_x: float = 0.0,
    force_y: float = 0.0,
    force_z: float = 0.0,
    torque_x: float = 0.0,
    torque_y: float = 0.0,
    torque_z: float = 0.0,
) -> trajectory_pb2.WrenchTrajectoryPoint:
    """
    Build a single trajectory point for a Wrench Trajectory
    :param seconds: seconds from start when this force should be reached
    :param force_x: force in x direction
    :param force_y: force in y direction
    :param force_z: force in z direction
    :param torque_x: torque around x-axis
    :param torque_y: torque around y-axis
    :param torque_z: torque around z-axis
    :return: trajectory_pb2.WrenchTrajectoryPoint
    """
    force = geometry_pb2.Vec3(x=force_x, y=force_y, z=force_z)
    torque = geometry_pb2.Vec3(x=torque_x, y=torque_y, z=torque_z)

    wrench = geometry_pb2.Wrench(force=force, torque=torque)
    t = seconds_to_duration(seconds)
    return trajectory_pb2.WrenchTrajectoryPoint(wrench=wrench, time_since_reference=t)
