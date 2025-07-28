"""
Xplore DS :: Encoding Datetime Variables
"""

import pandas as pd
from pathlib import Path
import sys

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))


def get_period(hour):
    if pd.isna(hour):
        return None
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 24:
        return "evening"
    else:
        return "night"


def build_relative_datetime_features(
    data: pd.DataFrame,
    variable_column_name: str,
    variable_reference_column_name: str = None,
    time_reference: str = "hours",
    log: object = None,
):

    if log:
        log.info(
            f"Encoding relative datetime features for {variable_column_name} using {time_reference} as reference time unit."
        )

    if time_reference not in [
        "seconds",
        "minutes",
        "hours",
        "days",
        "weeks",
        "months",
        "years",
    ]:
        raise ValueError(
            "Invalid time reference. Choose from 'seconds', 'minutes','hours', 'days', 'weeks', 'months' or 'years'."
        )
    if variable_column_name not in data.columns:
        raise ValueError(f"Column {variable_column_name} not found in DataFrame")
    if (
        variable_reference_column_name
        and variable_reference_column_name not in data.columns
    ):
        raise ValueError(
            f"Reference column {variable_reference_column_name} not found in DataFrame"
        )

    # Referencia de unidades de tempo
    if time_reference == "seconds":
        time_unit = 1
    elif time_reference == "minutes":
        time_unit = 60
    elif time_reference == "hours":
        time_unit = 3600
    elif time_reference == "days":
        time_unit = 86400
    elif time_reference == "weeks":
        time_unit = 604800
    elif time_reference == "months":
        time_unit = 2629800  # Aproximadamente 30 dias
    elif time_reference == "years":
        time_unit = 31557600  # Aproximadamente 365.25 dias

    # Tempo entre referencia e variavel
    data[
        "num_"
        + variable_column_name
        + "_relative_from_"
        + variable_reference_column_name
        + "_in_"
        + time_reference
    ] = (
        data[variable_column_name] - data[variable_reference_column_name]
    ).dt.total_seconds() / time_unit

    return data


def build_encoded_datetime_features(
    data: pd,
    variable_column_name: str,
    log: object = None,
):
    """
    Encodes datetime variables into multiple features.

    Args:
        data: Input DataFrame
        variable_column_name: Column to encode
        log: Optional logger instance

    Returns:
        DataFrame with encoded datetime features
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")

    if variable_column_name not in data.columns:
        raise ValueError(f"Column {variable_column_name} not found in DataFrame")

    if log:
        log.info(f"Encoding datetime features for {variable_column_name}.")

    try:

        data["cat_" + variable_column_name + "_year"] = data[
            variable_column_name
        ].dt.year
        data["cat_" + variable_column_name + "_month"] = data[
            variable_column_name
        ].dt.month
        data["cat_" + variable_column_name + "_day"] = data[variable_column_name].dt.day
        data["cat_" + variable_column_name + "_hour"] = data[
            variable_column_name
        ].dt.hour
        data["cat_" + variable_column_name + "_minute"] = data[
            variable_column_name
        ].dt.minute
        data["cat_" + variable_column_name + "_second"] = data[
            variable_column_name
        ].dt.second
        data["cat_" + variable_column_name + "_weekday"] = data[
            variable_column_name
        ].dt.weekday
        data["cat_" + variable_column_name + "_dayofyear"] = data[
            variable_column_name
        ].dt.dayofyear
        data["cat_" + variable_column_name + "_quarter"] = data[
            variable_column_name
        ].dt.quarter

        data["cat_" + variable_column_name + "_weekofmonth"] = (
            data[variable_column_name].dt.day - 1
        ) // 7 + 1

        data["cat_" + variable_column_name + "_period"] = data[
            variable_column_name
        ].dt.hour.apply(get_period)

        return data
    except Exception as e:
        if log:
            log.error(f"Error encoding datetime variable: {e}")
        raise e
