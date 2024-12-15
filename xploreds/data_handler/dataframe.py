"""
Xplore DS :: Dataframe Tools Package
"""

import pandas as pd


def describe_dataframe(data: pd, log=None) -> None:
    """
    Displays detailed information about a pandas DataFrame including memory usage and row counts.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to be described. Note that the type hint should ideally be
        pandas.DataFrame instead of just pd.

    log : Optional[Logger]
        A logger object to output the information. If None, no logging is performed.
        Default is None.
    """
    if log is not None:
        log.title("Dataframe description")
        log.info(data.info(verbose=True, memory_usage=True, show_counts=True))


def rename_columns(data: pd, columns_to_rename: dict, log=None) -> pd:
    """
    Renames specified columns in a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame whose columns are to be renamed.

    columns_to_rename : dict
        A dictionary where keys are the current column names and values are the new names.

    log : Optional[Logger]
        A logger object to output the information. If None, no logging is performed.
        Default is None.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with renamed columns.

    Raises
    ------
    KeyError
        If a column name in `columns_to_rename` does not exist in the DataFrame.
    """
    if log is not None:
        log.info("Renaming columns...")

    for old_name, new_name in columns_to_rename.items():
        if old_name not in data.columns:
            raise KeyError(f"Column '{old_name}' not found in DataFrame.")
        data = data.rename(columns={old_name: new_name})

    return data


def create_unique_id(data: pd, id_column_name: str = "id", log=None) -> pd:
    """
    Creates a unique identifier column for a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame for which the unique identifier column is to be created.

    id_column_name : str, optional
        The name of the new unique identifier column. Default is "id".

    log : Optional[Logger]
        A logger object to output the information. If None, no logging is performed.
        Default is None.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with the new unique identifier column.

    Raises
    ------
    ValueError
        If the DataFrame is empty.
    """
    if log is not None:
        log.info("Creating unique identifier column...")

    if data.empty:
        raise ValueError("DataFrame is empty.")

    data[id_column_name] = range(1, len(data) + 1)

    return data
