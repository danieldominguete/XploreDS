"""
Xplore DS :: Environment Utils Package

Utilities for locating project roots and resolving runtime paths.
"""

from pathlib import Path

PathLike = str | Path

DEFAULT_PROJECT_MARKER = "pyproject.toml"


def resolve_project_root(
    start: PathLike | None = None,
    *,
    marker: str = DEFAULT_PROJECT_MARKER,
) -> Path:
    """
    Locate a project root by walking parent directories for a marker file.

    Args:
        start: Directory to begin the search. Defaults to the current
            working directory when ``None``.
        marker: Filename that identifies the project root. Defaults to
            ``pyproject.toml``.

    Returns:
        Absolute path to the directory containing the marker file.

    Raises:
        RuntimeError: When ``marker`` cannot be found in any parent folder.

    Example:
        From a cookbook script::

            project_root = resolve_project_root(Path(__file__).resolve().parent)
    """
    current = Path(start).resolve() if start is not None else Path.cwd()

    for path in (current, *current.parents):
        if (path / marker).is_file():
            return path

    raise RuntimeError(
        f"Could not find {marker!r} starting from {current}. "
        "Run the script from within the project repository."
    )
