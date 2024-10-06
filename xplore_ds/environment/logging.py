"""
Xplore DS :: Logging Tools Package
"""

import logging
from datetime import datetime
import time
from pathlib import Path
from xplore_ds.data_handler.file import create_folder


class XploreDSLogging:

    def __init__(self, project_root: str, script_name: str) -> None:
        """
        Initialize the logging configuration for the project.

        This constructor sets up the logging environment by configuring file paths,
        creating necessary directories, and initializing the logger.

        Args:
            project_root (str): The root directory of the project.
            script_name (str): The name of the script being executed.

        Attributes:
            dt_init (datetime): Timestamp of logger initialization.
            dt_init_str (str): Formatted string of initialization timestamp.
            log_name (str): Name of the log file.
            log_run (str): Identifier for the current run.
            log_path (str): Directory path where logs will be stored.
            log_file (str): Full path to the log file.
            logger: Configured logging object.

        Returns:
            None

        Note:
            This method creates a new directory for each run under the 'runs' folder
            in the project root, named with the script name and timestamp.
        """

        # capturando referencias de tempos de inicialização
        self.dt_init = datetime.now()
        self.dt_init_str = self.dt_init.strftime("%y%m%d%H%M%S")
        self.ts_init = time.time()

        # configurando o nome do arquivo de log
        self.log_name = Path(str(script_name)).stem + "_" + self.dt_init_str + ".log"
        self.log_run = Path(str(script_name)).stem + "_" + self.dt_init_str

        # configurando locais de registros de logs
        self.log_path = str(project_root) + "/runs/" + str(self.log_run) + "/"

        # configurando o path completo de log
        self.log_file = self.log_path + self.log_name

        # criando a pasta caso não exista
        create_folder(self.log_path)

        # configurando o logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> None:
        """
        Set up and configure the logger for the class.

        This method configures the logging system with specific settings:
        - Sets the logging level to INFO
        - Defines a custom format for log messages
        - Sets up both file and stream handlers for logging

        The configuration includes:
        - Log level: INFO
        - Format: timestamp - log level - message
        - Date format: DD-MM-YYYY HH:MM:SS
        - Handlers: FileHandler (writes to a file) and StreamHandler (writes to console)

        Returns:
            None

        Note:
            This is an internal method, as indicated by the leading underscore.
            The log file path should be set in self.log_file before calling this method.
        """

        # configurando o logger
        log_config = {
            "level": logging.INFO,
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "datefmt": "%d-%m-%Y %H:%M:%S",
            "handlers": [logging.FileHandler(self.log_file), logging.StreamHandler()],
        }

        logging.basicConfig(**log_config)

        return logging.basicConfig(**log_config)

    def info(self, message: str) -> None:
        """
        Log an information-level message.

        This method logs the provided message using the logging module's info level.
        It serves as a wrapper around logging.info() within the context of the class.

        Args:
            message (str): The message to be logged.

        Returns:
            None

        Example:
            >>> self.info("Database connection established")
        """

        logging.info(message)

    def warning(self, message: str) -> None:
        """
        Log a warning-level message.

        This method logs the provided message using the logging module's warning level.
        It serves as a wrapper around logging.warning() within the context of the class.

        Args:
            message (str): The message to be logged as a warning.

        Returns:
            None

        Example:
            >>> self.warning("Low disk space detected")
        """
        logging.warning(message)

    def error(self, message: str) -> None:
        """
        Log an error-level message.

        This method logs the provided message using the logging module's error level.
        It serves as a wrapper around logging.error() within the context of the class.

        Args:
            message (str): The message to be logged as an error.

        Returns:
            None

        Example:
            >>> self.error("Database connection failed")
        """
        logging.error(message)

    def title(self, message: str) -> None:
        """
        Log an title message.

        Args:
            message (str): The message to be logged.

        Returns:
            None

        Example:
            >>> self.info("Database connection established")
        """
        logging.info(
            "=================================================================================="
        )
        logging.info(message)
        logging.info(
            "=================================================================================="
        )

    def init_run(self) -> None:
        """
        Initialize a new run in the log.

        This method logs the start of a new script execution by printing a
        distinctive header in the log file. It creates a visual separation
        between different runs, making it easier to identify the beginning
        of a new execution in the log.

        Returns:
            None


        """

        # configurando abertura de run
        logging.info(
            "=================================================================================="
        )
        logging.info("Starting script: " + self.log_run)
        logging.info(
            "=================================================================================="
        )

    def close_run(self) -> None:
        """
        Finalize and close the current run in the log.

        This method logs the end of the current script execution by printing a
        distinctive footer in the log file. It creates a visual separation to
        clearly mark the end of the current run, making it easier to identify
        where one execution ends in the log.

        """

        time_ref_end = datetime.now()
        time_end = time.time()
        exec_time = ((time_end - self.ts_init) / 60) / 60

        logging.info(
            "=================================================================================="
        )
        logging.info("Execution finished script: " + self.log_run)
        logging.info("Conclusion at " + str(time_ref_end.strftime("%Y-%m-%d %H:%M")))
        logging.info("Execution time: %.2f hours" % exec_time)
        logging.info(
            "=================================================================================="
        )
        logging.info(
            "                                                                          XploreDS"
        )
        logging.info(
            "=================================================================================="
        )
