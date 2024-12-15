"""
Xplore DS :: Missing Values Tools Package
"""

import os
import sys, os
from pathlib import Path
import pandas as pd
import numpy as np

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))


def normalize_not_valid_values(
    data: pd, not_valid_values: list = ["na", "N/A"], log=None
) -> pd:
    """
    Normalizes not valid values in a pandas DataFrame by replacing them with None.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to be normalized.

    log : Optional[Logger]
        A logger object to output the information. If None, no logging is performed.
        Default is None.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with normalized not valid values.

    Raises
    ------
    ValueError
        If the DataFrame is empty.
    """
    if log is not None:
        log.info("Normalizing not valid values...")

    if data.empty:
        raise ValueError("DataFrame is empty.")

    return data.replace(not_valid_values, None)


def delete_registers_with_missing(data: pd.DataFrame):

    # Removendo registros com algum valor nulo
    data = data.dropna(how="any", axis=0)

    return data


def replace_missing_values_by_default_value(
    data: pd.DataFrame,
    column_names: list = None,
    replacement_value: str = None,
    log=None,
):

    if log is not None:
        log.info("Replacing missing values by default value...")

    if column_names:
        # Substituindo valores nulos por um valor especifico
        data[column_names] = data[column_names].fillna(replacement_value)
    else:
        # Substituindo valores nulos por um valor especifico
        data = data.fillna(replacement_value)

    return data


def replace_missing_values_by_statistics_value(
    data: pd.DataFrame,
    column_names: list = None,
    replacement_value: str = "median",
    log=None,
):
    if log is not None:
        log.info("Replacing missing values by statistics value...")

    if replacement_value == "median":
        data[column_names] = data[column_names].fillna(data[column_names].median())
    elif replacement_value == "mean":
        data[column_names] = data[column_names].fillna(data[column_names].mean())
    elif replacement_value == "mode":
        data[column_names] = data[column_names].fillna(data[column_names].mode()[0])
    else:
        raise ValueError(
            "Invalid replacement value. Choose 'median', 'mean' or 'mode'."
        )

    return data
