import logging
import os.path

from utils.recursive_config import Config


class ProjectLogger(logging.Logger):
    def __init__(self, config: Config):
        super().__init__(name="ProjectLogger")
        self.config = config
        self.logs_folder = config.get_subpath("logs")
        self.base_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        self._setup_file_handler("debug.log", logging.DEBUG)
        self._setup_file_handler("info.log", logging.INFO)
        self._setup_file_handler("warning.log", logging.WARNING)
        self._setup_file_handler("error.log", logging.ERROR)
        self._setup_stream_handler(logging.INFO)

    def _setup_file_handler(self, file_name: str, level: int, format_str: str = None):
        path = os.path.join(self.logs_folder, file_name)
        handler = logging.FileHandler(path)
        handler.setLevel(level)
        if format_str is None:
            formatter = logging.Formatter(self.base_format)
        else:
            formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        self.addHandler(handler)
        return handler

    def _setup_stream_handler(self, level: int, format_str: str = None):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        if format_str is None:
            formatter = logging.Formatter(self.base_format)
        else:
            formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        self.addHandler(handler)
        return handler
