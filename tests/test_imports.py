"""Smoke tests for package imports and layout."""

import pandas as pd
import pytest

import xploreds
from xploreds.io.xds_file import create_folder, save_dataframe_to_parquet
from xploreds.log.xds_log import (
    XploreDSLogging,
    _configure_python_warnings,
    create_logger,
)
from xploreds.utils.xds_environment import resolve_project_root
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


def test_resolve_project_root_from_nested_folder(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    nested = tmp_path / "cookbook" / "01_data_mining"
    nested.mkdir(parents=True)
    assert resolve_project_root(nested) == tmp_path.resolve()


def test_resolve_project_root_raises_when_marker_missing(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="pyproject.toml"):
        resolve_project_root(tmp_path)


def test_create_logger(tmp_path) -> None:
    script_path = tmp_path / "pipeline.py"
    script_path.touch()
    logger = create_logger(tmp_path, script_path)
    assert isinstance(logger, XploreDSLogging)
    assert logger.script_name == "pipeline.py"
    assert logger.project_root == tmp_path


def test_save_dataframe_to_parquet_appends_extension(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2]})
    target = tmp_path / "dataset"
    save_dataframe_to_parquet(df, target)
    assert target.with_suffix(".parquet").is_file()


def test_save_dataframe_to_parquet_respects_overwrite(tmp_path) -> None:
    df = pd.DataFrame({"a": [1]})
    target = tmp_path / "dataset.parquet"
    save_dataframe_to_parquet(df, target)
    with pytest.raises(FileExistsError, match="already exists"):
        save_dataframe_to_parquet(df, target)
    save_dataframe_to_parquet(df, target, overwrite=True)
