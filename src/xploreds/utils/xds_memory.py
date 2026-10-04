"""
Xplore DS :: Memory Utils Package

Utilities for converting and expressing memory sizes in binary units.
"""

from typing import Final

BYTES_PER_UNIT: Final[int] = 1024
BINARY_UNITS: Final[tuple[str, ...]] = ("B", "KB", "MB", "GB", "TB")
_UNIT_EXPONENT: Final[dict[str, int]] = {
    unit: exponent for exponent, unit in enumerate(BINARY_UNITS)
}


def convert_bytes(size_bytes: int | float, to_unit: str = "GB") -> float:
    """
    Convert a byte count to the requested binary unit (KiB-based).

    Args:
        size_bytes: Size in bytes.
        to_unit: Target unit. One of B, KB, MB, GB, or TB (case-insensitive).

    Returns:
        Size expressed in the requested unit.

    Raises:
        ValueError: When ``to_unit`` is not supported.

    Example:
        ``convert_bytes(1073741824, "GB")`` returns ``1.0``.
    """
    unit = to_unit.upper()
    if unit not in _UNIT_EXPONENT:
        supported = ", ".join(BINARY_UNITS)
        raise ValueError(f"Invalid unit {to_unit!r}. Use one of: {supported}.")

    exponent = _UNIT_EXPONENT[unit]
    return float(size_bytes) / (BYTES_PER_UNIT**exponent)
