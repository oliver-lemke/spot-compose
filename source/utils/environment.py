from __future__ import annotations

import os

import openai
import yaml
from utils.recursive_config import Config


def _load_environment_file(config: Config) -> dict:
    env_path = config.get_subpath("environment")
    with open(env_path, encoding="UTF-8") as environment_file:
        environment_data = yaml.safe_load(environment_file)
    return environment_data


def set_api_keys(config: Config) -> None:
    api_data = _load_environment_file(config)["api"]
    # OpenAI
    openai_key = api_data["openai"]["key"]
    openai.api_key = openai_key


def set_robot_password(config: Config) -> None:
    env_config = get_environment_config(config, ["spot"])
    os.environ["BOSDYN_CLIENT_USERNAME"] = env_config["admin_username"]
    os.environ["BOSDYN_CLIENT_PASSWORD"] = env_config["admin_password"]


def get_environment_config(config: Config, config_keys: list[str]) -> dict:
    environment_data = _load_environment_file(config)
    for arg in config_keys:
        environment_data = environment_data[arg]
    return environment_data
