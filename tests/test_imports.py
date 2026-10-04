"""Smoke tests for package imports and layout."""

import xploreds
from xploreds.io.xds_file import create_folder
from xploreds.log.xds_log import XploreDSLogging, _configure_python_warnings
from xploreds.utils.xds_memory import convert_bytes


def test_version_is_defined() -> None:
    assert xploreds.__version__


def test_convert_bytes_to_gb() -> None:
    size = 1024**3
    assert convert_bytes(size, "GB") == 1.0


def test_create_folder_is_idempotent(tmp_path) -> None:
    folder = tmp_path / "artifacts"
    assert create_folder(folder) is True
    assert create_folder(folder) is False


def test_configure_python_warnings_defaults_without_env(monkeypatch) -> None:
    monkeypatch.delenv("PYTHON_WARNINGS", raising=False)
    _configure_python_warnings()


def test_xplore_ds_logging_without_env_file(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PYTHON_WARNINGS", raising=False)
    logger = XploreDSLogging(str(tmp_path), "test_script.py")
    assert logger.logger is not None
