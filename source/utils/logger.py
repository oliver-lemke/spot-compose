from __future__ import annotations

import logging
import os
from datetime import datetime

from utils.recursive_config import Config
from utils.singletons import _SingletonWrapper


class TimedFileLogger:
    def __init__(self, config: Config) -> None:
        """
        Initializes a new logger instance that logs messages to a file named with the current time.

        Parameters:
        - log_directory: The directory where the log file will be created.
        """
        self.config = config
        self.log_directory = config.get_subpath("logs")
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Sets up and configures the logger.

        Returns:
        - A configured Logger instance.
        """
        # Ensure the log directory exists
        os.makedirs(self.log_directory, exist_ok=True)

        # Generate a log file name using the current time
        log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
        log_path = os.path.join(self.log_directory, log_filename)

        # Create and configure logger
        logger = logging.getLogger(f"log_{log_filename}")
        logger.setLevel(logging.DEBUG)  # Set the minimum logged level to DEBUG

        # Avoid duplicate handlers if the logger already exists
        if not logger.handlers:
            # Create a file handler to log messages to the file
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)

            # Create a formatter and set it for the file handler
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)

            # Add the file handler to the logger
            logger.addHandler(file_handler)

        return logger

    def log(self, *messages: str, level: str = "info") -> None:
        """
        Logs a message with the specified level.

        Parameters:
        - level: The logging level ('info', 'debug', 'warning', 'error', 'critical').
        - message: The message to log.
        """
        for message in messages:
            if level.lower() == "info":
                self.logger.info(message)
            elif level.lower() == "debug":
                self.logger.debug(message)
            elif level.lower() == "warning":
                self.logger.warning(message)
            elif level.lower() == "error":
                self.logger.error(message)
            elif level.lower() == "critical":
                self.logger.critical(message)
            else:
                raise ValueError(f"Unsupported logging level: {level}")


class LoggerSingleton(_SingletonWrapper):
    """
    Singleton for Logger to allow for persistent storage and easy access.
    For more information on singleton see utils/singletons.py
    """

    _type_of_class = TimedFileLogger
