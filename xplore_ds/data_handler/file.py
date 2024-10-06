"""
Xplore DS :: File Tools Package
"""

import os
import sys, os
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))


def load_dataframe_from_csv(
    filepath: str,
    separator: str = ";",
    selected_columns: list = [],
    select_sample: str = "random",
    perc_sample: float = 1,
    order: str = "asc",
    log: object = None,
) -> pd:
    """
    Load a CSV database into a pandas DataFrame with various options for sampling and column selection.

    This function provides flexible options for loading CSV files, including partial loading
    with random or sequential sampling, and column selection.

    Args:
        filepath (str): Path to the CSV file.
        separator (str, optional): Delimiter used in the CSV file. Defaults to ";".
        selected_columns (list, optional): List of column names to load. If empty, all columns are loaded. Defaults to [].
        select_sample (str, optional): Method of sampling, either "random" or "sequential". Defaults to "random".
        perc_sample (float, optional): Percentage of rows to sample, between 0 and 1. Defaults to 1 (load all rows).
        order (str, optional): Order for sequential sampling, either "asc" or "desc". Defaults to "asc".
        log (object, optional): Logging object to use for output messages. If None, no logging is performed.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the loaded data.

    Raises:
        ValueError: If invalid arguments are provided.

    Note:
        - The function first counts the total number of rows in the file to calculate the sample size.
        - For partial loading (perc_sample < 1), it uses numpy to generate indices for random sampling,
          or a list comprehension for sequential sampling.
        - The function uses pandas read_csv for actual data loading, with different parameters based on
          whether it's a partial or full load, and whether specific columns are selected.
        - If a logging object is provided, it logs information about the loaded dataset.

    """

    # number of rows from file without header
    num_lines = sum(1 for l in open(filepath)) - 1

    # sample lines size
    num_lines_selected = int(perc_sample * num_lines)
    # skip_lines = num_lines - num_lines_selected

    log.info("Total samples: {a:.1f}".format(a=num_lines))
    log.info("Total samples target: {a:.1f}".format(a=num_lines_selected))

    # Partial loading
    if perc_sample < 1:
        lines2skip = []
        if select_sample == "random":
            lines2skip = np.random.choice(
                np.arange(1, num_lines + 1),
                (num_lines - num_lines_selected),
                replace=False,
            )

        if select_sample == "sequential":
            if order == "asc":
                lines2skip = [
                    x for x in range(1, num_lines + 1) if x > num_lines_selected
                ]

            if order == "desc":
                lines2skip = [
                    x
                    for x in range(1, num_lines + 1)
                    if x <= (num_lines - num_lines_selected)
                ]

        if len(selected_columns) > 0:
            df = pd.read_csv(
                filepath,
                header=0,
                sep=separator,
                usecols=selected_columns,
                skiprows=lines2skip,
                encoding="utf-8",
                quotechar='"',
                escapechar="\\",
                low_memory=True,
                # engine='python',
                # quoting=csv.QUOTE_NONE,
                skipinitialspace=True,
            )
        else:
            df = pd.read_csv(
                filepath,
                header=0,
                sep=separator,
                skiprows=lines2skip,
                encoding="utf-8",
                quotechar='"',
                escapechar="\\",
                low_memory=True,
                # quoting=csv.QUOTE_NONE,
                skipinitialspace=True,
            )
    # Integral loading
    else:
        if len(selected_columns) > 0:
            df = pd.read_csv(
                filepath,
                header=0,
                sep=separator,
                usecols=selected_columns,
                encoding="utf-8",
                quotechar='"',
                escapechar="\\",
                low_memory=True,
                # quoting=csv.QUOTE_NONE,
                skipinitialspace=True,
            )
        else:
            df = pd.read_csv(
                filepath,
                header=0,
                sep=separator,
                encoding="utf-8",
                quotechar='"',
                escapechar="\\",
                low_memory=True,
                skipinitialspace=True,
            )

    log.info("Selected dataset samples: {a:.1f}".format(a=df.shape[0]))
    log.info("Number of variables: {a:.1f}".format(a=df.shape[1]))
    log.info("Variables list: {a:s}".format(a=str(df.columns.values.tolist())))

    return df


def create_folder(folder_path: str) -> bool:
    """
    Create a new folder at the specified path if it doesn't already exist.

    This function attempts to create a new directory at the given path.
    If the directory already exists, the function will not create a new one.

    Args:
        folder_path (str): The path where the new folder should be created.

    Returns:
        bool: True if a new folder was created, None otherwise.

    Note:
        - This function uses os.path.exists() to check if the folder already exists.
        - It uses os.makedirs() to create the new directory.
        - If the folder already exists, the function will not raise an error,
          it will simply return None.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return True

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return True


def get_name_and_extension_from_file(filename: str) -> list:
    """
    Split a filename into its name and extension.

    This function takes a filename as input and returns a list containing
    the file name (without extension) and the file extension.

    Args:
        filename (str): The full name of the file, including its extension.

    Returns:
        list: A list containing two elements:
            - The file name without the extension (str)
            - The file extension including the dot (str)
                If there's no extension, this will be an empty string.



    Note:
        This function uses os.path.splitext() to split the filename.
        It handles filenames with multiple dots correctly, considering
        only the last dot as the extension separator.
    """
    return os.path.splitext(filename)


def get_filename_from_path(path: str) -> str:
    """
    Extract the filename from a given file path.

    This function takes a file path as input and returns just the filename
    (including the extension) without the preceding directory path.

    Args:
        path (str): The full path to the file, including the filename.

    Returns:
        str: The filename extracted from the path, including its extension.


    Note:
        - This function uses Path from the pathlib module to handle the path.
        - It works with both Unix-style and Windows-style path separators.
        - If the path ends with a directory separator, it will return an empty string.
    """
    from pathlib import Path

    return Path(path).name


def load_parameters_from_file(path_file: str) -> dict:
    """
    Load parameters from a JSON file.

    This function attempts to open and read a JSON file at the specified path,
    and return its contents as a Python dictionary.

    Args:
        path_file (str): The path to the JSON file containing the parameters.

    Returns:
        dict: A dictionary containing the parameters loaded from the JSON file.

    Raises:
        Exception: If there's an error opening or parsing the JSON file.
            The specific exception type depends on what went wrong
            (e.g., FileNotFoundError, json.JSONDecodeError).

    """
    try:
        with open(path_file, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {path_file} was not found.")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {path_file}: {e}", e.doc, e.pos)
    except PermissionError:
        raise PermissionError(f"Permission denied when trying to read {path_file}")
    except IOError as e:
        raise IOError(f"I/O error occurred when reading {path_file}: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")
    return data


def get_nrows_from_file(filepath) -> int:
    """
    Get the number of rows in a file.

    This function opens a file at the specified path and counts the number of lines
    (rows) in the file. It's useful for quickly determining the size of a dataset
    or file without loading the entire file into memory.

    Args:
        filepath (str): The path to the file whose rows are to be counted.

    Returns:
        int: The number of rows in the file.

    Raises:
        IOError: If there's an error opening or reading the file.
            The specific exception type depends on what went wrong
            (e.g., FileNotFoundError, PermissionError).

    Note:
        - This function uses a generator expression with enumerate to efficiently
          count the lines in the file.
        - It's particularly useful for large files where counting all lines at once
          might be impractical due to memory constraints.
    """
    with open(filepath) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def save_dataframe_to_parquet(data: pd, file_path: str, log: object) -> None:
    """
    Save a pandas DataFrame to a Parquet file.

    This function takes a pandas DataFrame and a file path, and saves the DataFrame
    to a Parquet file at the specified path. It's useful for efficiently storing
    large datasets in a columnar format.

    Args:
        data (pandas.DataFrame): The DataFrame to save.
        file_path (str): The path where the Parquet file will be saved.

    Returns:
        None: This function does not return anything. It saves the DataFrame to a file.

    Raises:
        IOError: If there's an error opening or writing to the file.
            The specific exception type depends on what went wrong
            (e.g., FileNotFoundError, PermissionError).

    Note:
        - This function uses the pandas to_parquet method to save the DataFrame.
        - It's particularly useful for large datasets where the Parquet format
          provides efficient compression and columnar storage.
    """
    log.info("Saving dataframe to parquet file: " + file_path)

    # verificando se a pasta existe caso contrario criar a pasta
    create_folder(os.path.dirname(file_path))

    data.to_parquet(file_path)


def load_dataframe_from_parquet(file_path: str, log: object) -> pd:
    """
    Load a pandas DataFrame from a Parquet file.

    This function takes a file path, and loads the DataFrame
    from a Parquet file at the specified path.

    Args:
        file_path (str): The path where the Parquet file will be saved.

    Returns:
        pandas.DataFrame: The DataFrame loaded from the Parquet file.

    Raises:
        IOError: If there's an error opening or writing to the file.
            The specific exception type depends on what went wrong
            (e.g., FileNotFoundError, PermissionError).

    Note:
        - This function uses the pandas read_parquet method to load the DataFrame.
        - It's particularly useful for large datasets where the Parquet format
          provides efficient compression and columnar storage.
    """
    log.info("Loading dataframe from parquet file: " + file_path)

    return pd.read_parquet(file_path)
