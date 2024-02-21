"""
This utility file contains several Singleton classes, designed to ensure that only one instance of each class exists
throughout the application. These singletons are used to manage shared resources and global state in a controlled
manner.

This allows us to have access to the objects the singletons represent from anywhere in the code, without always having
to give them as variables.

I.e. instead of calling
method(..., [singleton]) every time, we can call
singleton = Singleton() once at the beginning of every file that needs access to it.
It will be automatically instantiated everywhere, once it is instantiated anywhere.
"""

from __future__ import annotations

from typing import Iterable

import bosdyn


class SingletonNotInstantiatedException(Exception):
    pass


class WrongWrappedObjectException(Exception):
    pass


class ProhibitedSingletonOverwriteException(Exception):
    pass


class _Singleton:
    def __init__(self, type_of_class: type, allow_overwrite: bool = True):
        self._instance = None
        self._type_of_class = type_of_class
        self._is_instantiated = False
        self._allow_overwrite = allow_overwrite

    def set_instance(self, instance):
        if not isinstance(instance, self._type_of_class):
            raise WrongWrappedObjectException(
                f"Wrapped object must be of type {self._type_of_class}!"
            )
        if self._instance is not None and not self._allow_overwrite:
            raise ProhibitedSingletonOverwriteException(
                "Cannot overwrite singleton wrapped value!"
            )
        self._instance = instance
        self._is_instantiated = True

    def is_instantiated(self):
        return self._is_instantiated

    def reset(self):
        self._instance = None
        self._is_instantiated = False

    def __getattr__(self, name):
        if not self._is_instantiated:
            raise SingletonNotInstantiatedException("Singleton was never instantiated!")
        return getattr(self._instance, name)


class _SingletonWrapper:
    _instance = None
    _type_of_class = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = _Singleton(cls._type_of_class)
        return cls._instance

    def __getattr__(self, name):
        return getattr(self._instance, name)


class RobotSingleton(_SingletonWrapper):
    _type_of_class = bosdyn.client.Robot


class RobotCommandClientSingleton(_SingletonWrapper):
    _type_of_class = bosdyn.client.robot_command.RobotCommandClient


class RobotStateClientSingleton(_SingletonWrapper):
    _type_of_class = bosdyn.client.robot_state.RobotStateClient


class WorldObjectClientSingleton(_SingletonWrapper):
    _type_of_class = bosdyn.client.world_object.WorldObjectClient


class ImageClientSingleton(_SingletonWrapper):
    _type_of_class = bosdyn.client.image.ImageClient


class GraphNavClientSingleton(_SingletonWrapper):
    _type_of_class = bosdyn.client.graph_nav.GraphNavClient


def reset_singletons(singletons: Iterable[_Singleton]):
    for singleton in singletons:
        singleton.reset()


# from robot_utils.frame_transformer import FrameTransformerSingleton
# from utils.singletons import (
#     GraphNavClientSingleton,
#     ImageClientSingleton,
#     RobotCommandClientSingleton,
#     RobotSingleton,
#     RobotStateClientSingleton,
#     WorldObjectClientSingleton,
# )
#
# frame_transformer = FrameTransformerSingleton()
# graph_nav_client = GraphNavClientSingleton()
# image_client = ImageClientSingleton()
# robot_command_client = RobotCommandClientSingleton()
# robot = RobotSingleton()
# robot_state_client = RobotStateClientSingleton()
# world_object_client = WorldObjectClientSingleton()
