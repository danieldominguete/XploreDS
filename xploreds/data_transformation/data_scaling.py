"""
Xplore DS :: Scaling Variables
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path
import sys

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xploreds.data_schemas.model_io_config import ScalingMethod


def scaler_variable_fit(
    data: pd, variable_column_name: str, scale_method: ScalingMethod, log: object = None
):
    """
    Fits a scaler on the specified column data.

    Args:
        data: Input DataFrame
        variable_column_name: Column to scale
        scale_method: Scaling method to use
        log: Optional logger instance

    Returns:
        Fitted scaler instance
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")

    if variable_column_name not in data.columns:
        raise ValueError(f"Column {variable_column_name} not found in DataFrame")

    if scale_method == ScalingMethod.mean_std_scaler:
        scaler = StandardScaler()
    elif scale_method == ScalingMethod.min_max_scaler:
        scaler = MinMaxScaler()
    else:
        scaler = None
        return scaler

    try:
        if log:
            log.info(f"Fitting scaler {scale_method} of {variable_column_name} ...")
        scaler.fit(data[variable_column_name].values.reshape(-1, 1))
        return scaler
    except Exception as e:
        if log:
            log.error(f"Error fitting scaler: {str(e)}")
        raise


def scaler_variable_transform(
    data: pd,
    variable_column_name: str,
    scaler,
    log: object = None,
):
    """
    Transforms data using the fitted scaler.

    Args:
        data: Input DataFrame
        feature_column_name: Column to scale
        scaler: Fitted scaler instance
        log: Optional logger instance

    Returns:
        Tuple containing:
        - Transformed DataFrame
        - Name of the scaled column
    """
    scaled_column_name = f"{variable_column_name}_scaled"

    try:
        if scaler:
            data[variable_column_name + "_scaled"] = scaler.transform(
                data[variable_column_name].values.reshape(-1, 1)
            )
        else:
            data[variable_column_name + "_scaled"] = data[variable_column_name]

        return data, scaled_column_name

    except Exception as e:
        if log:
            log.error(f"Error transforming data: {str(e)}")
        raise


def scaler_variable_fit_transform(
    data: pd,
    variable_column_name: str,
    scale_method: ScalingMethod,
    log: object = None,
):
    """
    Combines fitting and transformation in one step.

    Args:
        data: Input DataFrame
        variable_column_name: Column to scale
        scale_method: Scaling method to use
        log: Optional logger instance

    Returns:
        Tuple containing:
        - Transformed DataFrame with new scaled column
        - Name of the scaled column
    """

    scaler = scaler_variable_fit(
        data=data,
        variable_column_name=variable_column_name,
        scale_method=scale_method,
        log=log,
    )

    data, scaled_variable = scaler_variable_transform(
        data=data,
        variable_column_name=variable_column_name,
        scaler=scaler,
        log=log,
    )

    return data, scaled_variable
