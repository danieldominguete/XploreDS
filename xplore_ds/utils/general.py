def convert_bytes(bytes: int, to_unit: str = "GB") -> float:
    units = {"KB": 1, "MB": 2, "GB": 3, "TB": 4}
    if to_unit not in units:
        raise ValueError("Invalid unit. Use 'KB', 'MB', 'GB', or 'TB'")
    return float(bytes / (1024 ** units[to_unit]))
