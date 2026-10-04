f"""
Xplore DS :: I/O Main Package

Utilities for folder management, JSON I/O, and pandas DataFrame persistence.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

PathLike = Union[str, Path]


def _as_path(path: PathLike) -> Path:
    """Normalize a path-like value to ``Path``."""
    return Path(path)


def _ensure_parent_dir(file_path: PathLike) -> None:
    """Create the parent directory of a file path when it does not exist."""
    parent = _as_path(file_path).parent
    # Ignora caminhos relativos na raiz (ex.: "arquivo.json")
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


def _log_dataframe_summary(logger: logging.Logger, data: pd.DataFrame) -> None:
    """Log shape and column names of a DataFrame."""
    logger.info("Total samples: %s", data.shape[0])
    logger.info("Number of variables: %s", data.shape[1])
    logger.info("Variables list: %s", data.columns.tolist())


def create_folder(folder_path: PathLike) -> bool:
    """
    Create a folder at the specified path when it does not exist.

    Args:
        folder_path: Directory path to create. Accepts ``str`` or ``Path``.

    Returns:
        True when a new folder is created, False when it already exists.

    Note:
        Nested directories are created automatically (``parents=True``).
    """
    path = _as_path(folder_path)

    # Retorna cedo quando o diretorio ja existe
    if path.exists():
        return False

    path.mkdir(parents=True, exist_ok=True)
    return True


def get_name_and_extension_from_file(filename: str) -> tuple[str, str]:
    """
    Split a filename into its base name and extension.

    Args:
        filename: Full filename or path containing the file name.

    Returns:
        A tuple ``(name, extension)``. Extension includes the leading dot.
        When there is no extension, the second element is an empty string.

    Example:
        ``get_name_and_extension_from_file("archive.tar.gz")`` returns
        ``("archive.tar", ".gz")``.
    """
    path = Path(Path(filename).name)
    return path.stem, path.suffix


def get_filename_from_path(path: PathLike) -> str:
    """
    Extract the filename from a file path.

    Args:
        path: Full path to the file. Accepts ``str`` or ``Path``.

    Returns:
        Filename including extension, without parent directories.
    """
    return _as_path(path).name


def load_dictionary_from_json(path_file: PathLike) -> dict[str, Any]:
    """
    Load a dictionary from a JSON file.

    Args:
        path_file: Path to the JSON file. Accepts ``str`` or ``Path``.

    Returns:
        Dictionary parsed from the JSON content.

    Raises:
        FileNotFoundError: When the file does not exist.
        json.JSONDecodeError: When the file content is not valid JSON.
        PermissionError: When the file cannot be read due to permissions.
    """
    path = _as_path(path_file)
    if not path.is_file():
        raise FileNotFoundError(f"The file {path} was not found.")

    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def save_dictionary_to_json(
    data: dict[str, Any],
    file_path: PathLike,
    log: Optional[logging.Logger] = None,
) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        data: Dictionary to serialize.
        file_path: Destination file path. Accepts ``str`` or ``Path``.
        log: Optional logger for progress messages.

    Raises:
        OSError: When the file cannot be written.
        TypeError: When ``data`` is not JSON-serializable.

    Note:
        Creates parent directories automatically when they do not exist.
    """
    path = _as_path(file_path)

    if log:
        log.info("Saving dictionary to json file...")

    # Garante diretorio pai antes de gravar
    _ensure_parent_dir(path)

    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)

    if log:
        log.info("Dictionary saved to json file: %s", path)


def get_nrows_from_file(filepath: PathLike) -> int:
    """
    Count the number of lines in a text file.

    Args:
        filepath: Path to the text file. Accepts ``str`` or ``Path``.

    Returns:
        Number of lines in the file. Returns ``0`` for empty files.

    Raises:
        OSError: When the file cannot be opened or read.
    """
    with _as_path(filepath).open(encoding="utf-8") as file:
        return sum(1 for _ in file)


def save_dataframe_to_parquet(
    data: pd.DataFrame,
    file_path: PathLike,
    log: Optional[logging.Logger] = None,
) -> None:
    """
    Save a pandas DataFrame to a Parquet file.

    Args:
        data: DataFrame to persist.
        file_path: Destination file path. Accepts ``str`` or ``Path``.
        log: Optional logger for progress and dataset summary.

    Raises:
        OSError: When the file cannot be written.

    Note:
        Requires ``pyarrow`` or ``fastparquet`` installed in the environment.
    """
    path = _as_path(file_path)

    if log:
        log.info("Saving dataframe to parquet file...")

    _ensure_parent_dir(path)
    data.to_parquet(path)

    if log:
        log.info("Dataframe saved to parquet file: %s", path)
        _log_dataframe_summary(log, data)


def load_dataframe_from_parquet(
    file_path: PathLike,
    selected_columns: Optional[Sequence[str]] = None,
    log: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Load a pandas DataFrame from a Parquet file.

    Args:
        file_path: Path to the Parquet file. Accepts ``str`` or ``Path``.
        selected_columns: Optional column subset to load. Loads all columns
            when ``None``.
        log: Optional logger for progress and dataset summary.

    Returns:
        DataFrame loaded from the Parquet file.

    Raises:
        OSError: When the file cannot be read.
    """
    path = _as_path(file_path)
    columns = list(selected_columns) if selected_columns else None
    data = pd.read_parquet(path, columns=columns)

    if log:
        log.info("Dataframe loaded from parquet file: %s", path)
        _log_dataframe_summary(log, data)

    return data


def save_dataframe_to_excel(
    data: pd.DataFrame,
    file_path: PathLike,
    sheet_name: str = "data",
    log: Optional[logging.Logger] = None,
) -> None:
    """
    Save a pandas DataFrame to an Excel file.

    Args:
        data: DataFrame to persist.
        file_path: Destination file path. Accepts ``str`` or ``Path``.
        sheet_name: Worksheet name. Defaults to ``"data"``.
        log: Optional logger for progress and dataset summary.

    Raises:
        OSError: When the file cannot be written.

    Note:
        Requires ``openpyxl`` installed in the environment.
        The index is not written to the spreadsheet.
    """
    path = _as_path(file_path)

    if log:
        log.info("Saving dataframe to excel file...")

    _ensure_parent_dir(path)
    data.to_excel(path, sheet_name=sheet_name, index=False)

    if log:
        log.info("Dataframe saved to excel file: %s", path)
        _log_dataframe_summary(log, data)
