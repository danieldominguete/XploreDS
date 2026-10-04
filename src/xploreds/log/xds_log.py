"""
Xplore DS :: Logging Tools Package

Structured logging for scripts and cookbook runs, including environment profiling.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Final

import psutil

from xploreds.io.xds_file import create_folder
from xploreds.utils.xds_memory import convert_bytes

PathLike = str | Path

_LOGGER_NAME: Final = "xploreds"
_LOG_FORMAT: Final = "%(asctime)s - %(levelname)s - %(message)s"
_LOG_DATE_FORMAT: Final = "%d-%m-%Y %H:%M:%S"
_RUN_TIMESTAMP_FORMAT: Final = "%y%m%d%H%M%S"

_SEPARATOR_TITLE: Final = "*" * 82
_SEPARATOR_SUBTITLE: Final = "=" * 82
_SEPARATOR_SECTION: Final = "-" * 82

_VALID_WARNING_ACTIONS: Final = frozenset(
    {"error", "ignore", "always", "default", "module", "once"}
)


def _configure_python_warnings() -> None:
    """Apply warning filter from ``PYTHON_WARNINGS`` env var (default: ``default``)."""
    action = os.getenv("PYTHON_WARNINGS", "default")
    if action not in _VALID_WARNING_ACTIONS:
        action = "default"
    warnings.filterwarnings(action)


class _FlushingStreamHandler(logging.StreamHandler):
    """Stream handler that flushes after each record (helps IDE terminals)."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()
        stream = self.stream
        if stream is not None and hasattr(stream, "flush"):
            stream.flush()


def _log_bordered(logger: logging.Logger, message: str, separator: str) -> None:
    """Log a message framed by the same separator line above and below."""
    logger.info(separator)
    logger.info(message)
    logger.info(separator)


def _console_stream():
    """Return the real stdout stream, bypassing IDE redirects when possible."""
    return sys.__stdout__ if sys.__stdout__ is not None else sys.stdout


class XploreDSLogging:
    """
    Configure and write structured logs for a single script execution.

    Each run creates a timestamped folder under ``runs/`` with a dedicated
    ``.log`` file. Use wrapper methods (``info``, ``title``, ``section``) to
    keep log output consistent across projects.

    Args:
        project_root: Root directory of the project.
        script_name: Name or path of the script being executed (used in log paths).
    """

    def __init__(self, project_root: PathLike, script_name: PathLike) -> None:
        self.project_root = Path(project_root)
        script_path = Path(script_name)
        self.script_name = script_path.name

        self.dt_init = datetime.now()
        self.dt_init_str = self.dt_init.strftime(_RUN_TIMESTAMP_FORMAT)
        self.ts_init = time.time()

        script_stem = script_path.stem
        self.log_run = f"{script_stem}_{self.dt_init_str}"
        self.log_path = self.project_root / "runs" / self.log_run
        self.log_name = f"{self.log_run}.log"
        self.log_file = self.log_path / self.log_name

        create_folder(self.log_path)
        self.logger = self._setup_logger()
        _configure_python_warnings()

    def _setup_logger(self) -> logging.Logger:
        """Configure file and console handlers for the current run."""
        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False

        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
        file_handler = logging.FileHandler(self.log_file)
        stream = _console_stream()
        stream_handler = _FlushingStreamHandler(stream)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)

        for handler in (file_handler, stream_handler):
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def info(self, message: str) -> None:
        """Log an information-level message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning-level message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error-level message."""
        self.logger.error(message)

    def title(self, message: str) -> None:
        """Log a top-level title with a heavy separator."""
        _log_bordered(self.logger, message, _SEPARATOR_TITLE)

    def subtitle(self, message: str) -> None:
        """Log a subtitle with a medium separator."""
        _log_bordered(self.logger, message, _SEPARATOR_SUBTITLE)

    def section(self, message: str) -> None:
        """Log a section header with a light separator."""
        _log_bordered(self.logger, message, _SEPARATOR_SECTION)

    def init_run(self) -> None:
        """Log the start banner for the current run."""
        _log_bordered(self.logger, f"Starting script: {self.log_run}", _SEPARATOR_TITLE)

    def close_run(self) -> None:
        """Log the closing banner with execution duration."""
        ended_at = datetime.now()
        exec_hours = (time.time() - self.ts_init) / 3600

        self.logger.info(_SEPARATOR_TITLE)
        self.logger.info("Execution finished script: %s", self.log_run)
        self.logger.info("Conclusion at %s", ended_at.strftime("%Y-%m-%d %H:%M"))
        self.logger.info("Execution time: %.2f hours", exec_hours)
        self.logger.info(_SEPARATOR_TITLE)
        self.logger.info("%74sXploreDS", "")
        self.logger.info(_SEPARATOR_TITLE)

    def log_environment_setup(self) -> None:
        """
        Log hardware, OS, Python, and run configuration details.

        Note:
            Requires ``psutil`` for memory and CPU reporting.
        """
        self.logger.info(_SEPARATOR_TITLE)
        self.logger.info("Environment Setup")
        self._log_machine_setup()
        self._log_os_setup()
        self._log_python_setup()
        self._log_run_setup()
        self.logger.info(_SEPARATOR_TITLE)

    def _log_section(self, title: str) -> None:
        """Log a labeled section with light separators."""
        _log_bordered(self.logger, title, _SEPARATOR_SECTION)

    def _log_machine_setup(self) -> None:
        """Log CPU and memory details."""
        self._log_section("Machine Setup")
        self.logger.info("Machine: %s", platform.machine())
        self.logger.info("Processor: %s", platform.processor())
        self.logger.info("Physical cores: %s", psutil.cpu_count(logical=False))
        self.logger.info("Total cores: %s", psutil.cpu_count(logical=True))
        memory_gb = convert_bytes(psutil.virtual_memory().total, "GB")
        self.logger.info("Physical memory: %.2fGB", memory_gb)

    def _log_os_setup(self) -> None:
        """Log operating system details."""
        self._log_section("OS Setup")
        self.logger.info("OS id: %s", platform.platform())
        self.logger.info("OS System: %s", platform.system())
        self.logger.info("OS Node Name: %s", platform.node())
        self.logger.info("OS Release: %s", platform.release())

    def _log_python_setup(self) -> None:
        """Log Python interpreter details."""
        self._log_section("Python Setup")
        self.logger.info("Python version: %s", sys.version)
        self.logger.info("Python path: %s", sys.executable)
        self.logger.info("Python warnings: %s", os.getenv("PYTHON_WARNINGS"))

    def _log_run_setup(self) -> None:
        """Log paths and identifiers for the current run."""
        self._log_section("Run Setup")
        self.logger.info("Root folder: %s", self.project_root)
        self.logger.info("Run name: %s", self.log_run)
        self.logger.info("Artifacts folder: %s", self.log_path)
