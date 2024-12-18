import pandas as pd

"""
Xplore DS :: Date Time Tools Package
"""
import sys, os
from pathlib import Path
import pandas as pd

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))


def create_past_datetime_mask_from_reference(
    data: pd.DataFrame,
    id_entity_column_name: str,
    t0_reference_column_name: str,
    time_step_unit: str,
    time_step_amount: int,
    log: object = None,
) -> pd.DataFrame:
    """
    Create a mask for past datetime values based on a reference date.

    This function generates a mask for past datetime values based on a reference date.
    It can be used to filter data for past time periods.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        id_entity_column_name (str): The name of the column containing the entity IDs.
        t0_reference_column_name (str): The name of the column containing the reference date.
        time_step_unit (str): The unit of time step (e.g., 'days', 'months', 'years').
        time_step_amount (int): The amount of time step.
        log (object, optional): A logging object for logging messages. Defaults to None.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'past_datetime_mask' containing the mask for past datetime values.

    Raises:
        ValueError: If the specified time step unit is not supported.

    Note:
        - The function assumes that the reference date column is in datetime format.
        - The function uses the pandas library for datetime operations.
    """

    if log:
        log.info("Creating past datetime mask from reference...")

    # Create a copy of the input DataFrame
    df = data[[id_entity_column_name, t0_reference_column_name]]

    # Create a mask for past datetime values from t0_reference
    for step in range(1, time_step_amount + 1):
        df[f"dt_{step}"] = df[t0_reference_column_name] - pd.DateOffset(
            **{time_step_unit: step}
        )

    # Melt the dt_{step} columns
    melted_df = pd.melt(
        df,
        id_vars=[id_entity_column_name, t0_reference_column_name],
        value_vars=[f"dt_{step}" for step in range(1, time_step_amount + 1)],
        var_name="step_id",
        value_name="step_datetime",
    )

    # Clean up the step column to contain only the number
    melted_df["step_id"] = melted_df["step_id"].str.extract("(\d+)").astype(int)

    if log:
        log.info("Past datetime mask created from reference.")

    return melted_df
