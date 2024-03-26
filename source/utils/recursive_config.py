from __future__ import annotations

import datetime
import os

import yaml


class Config:
    """
    Config class that merges base and personal config file. Has some useful functions for ease of use.
    In the __init__ you can specify the lowest config file. By default, this is "user.yaml".
    Each .yaml file (except the top) has an extends attribute, which has the name of another .yaml file.
    The config class will search for that .yaml file, and overwrite its values with the ones we already have.
    By default, we have two files: config.yaml and user.yaml, which extends the former.
    That means user.yaml overwrites values in config.yaml. The idea of this configuration is that you have default values
    in config.yaml, and any custom attributes we want to overwrite on the local machine, you can store in user.yaml.
    """

    def __init__(self, file=None):
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

        if file is None:
            file = "config.yaml"
        else:
            file = f"{file}.yaml"

        def load_recursive(config: str, stack: list[str]) -> dict:
            """
            Load .yaml files recursively.
            """
            if config in stack:
                raise AssertionError("Attempting to build recursive configuration.")

            # load the .yaml file as a dict
            config_path = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            config_path = os.path.join(config_path, "configs", config)
            with open(config_path, "r", encoding="UTF-8") as file_handle:
                cfg = yaml.safe_load(file_handle)

            # check if the config has an extends attribute
            # if not, it is the base (highest) config file, which all others overwrite
            base = (
                {}
                if "extends" not in cfg
                else load_recursive(cfg["extends"], stack + [config])
            )
            # all the higher config files have been loaded
            # overwrite the values in these higher configs with the ones we have currently
            base = _recursive_update(base, cfg)
            return base

        # start config loading from the file specified in the parameter (user.yaml by default)
        self._config = load_recursive(file, [])

        if "project_root_dir" not in self._config:
            cwd = os.getcwd()
            project_root_dir = os.path.dirname(cwd)
            self._config["project_root_dir"] = project_root_dir

        self._add_additional_info()

    def _add_additional_info(self) -> None:
        additions = {}

        # git hash
        # repo = git.Repo(search_parent_directories=True)
        # sha = repo.head.object.hexsha
        # additions["git_sha"] = sha

        # current timestamp
        additions["timestamp"] = self.timestamp

        self._add_metadata(additions)

    def _add_metadata(self, additions: dict) -> None:
        if "metadata" not in self._config:
            self._config["metadata"] = {}

        for k, v in additions.items():
            if k in self._config["metadata"]:
                raise ValueError(
                    f"Please do not specify a {k} field in the metadata field of the "
                    f"config, as it is later added in post-processing"
                )
            else:
                self._config["metadata"][k] = v

    def get_subpath(self, subpath: str) -> str:
        """
        In config, there is one key called "subpaths".
        This method automatically combines the base path specified in "project_root_dir" with the specified subpath.
        """
        subpath_dict = self._config["subpaths"]
        if subpath not in list(subpath_dict.keys()):
            raise ValueError(f"Subpath {subpath} not known.")
        base_path = os.path.normpath(self._config["project_root_dir"])
        # return absolute path is that is specified, else concat with root dir
        if os.path.isabs(subpath):
            return os.path.normpath(subpath)
        path_ending = os.path.normpath(subpath_dict[subpath])
        return os.path.join(base_path, path_ending)

    def __getitem__(self, item):
        return self._config.__getitem__(item)

    def __setitem__(self, key, value):
        return self._config.__setitem__(key, value)

    def get(self, key, default=None):
        self._config.get(key, default=default)

    def get_config(self):
        return self._config


def _recursive_update(base: dict, cfg: dict) -> dict:
    for k, v in cfg.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = _recursive_update(base[k], v)
        else:
            base[k] = v
    return base
