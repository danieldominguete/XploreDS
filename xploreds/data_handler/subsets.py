"""
Xplore DS :: Dataset Tools Package
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_test_data_subsets(
    data: pd,
    proportion_test_samples: float = 0.1,
    random_state: int = None,
    shuffle: bool = False,
    log: object = None,
) -> pd:
    """
    Splits a dataset into training and test subsets using scikit-learn's train_test_split.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame to be split into training and test sets.

    proportion_test_samples : float, optional
        The proportion of the dataset to include in the test split.
        Should be between 0.0 and 1.0. Default is 0.1 (10% for testing).

    random_state : int, optional
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        Default is None.

    shuffle : bool, optional
        Whether or not to shuffle the data before splitting.
        Default is False.

    log : object, optional
        Logger object to record the splitting information.
        If None, no logging will be performed.
        Default is None.

    Returns
    -------
    tuple
        A tuple containing:
        - data_train (pandas.DataFrame): The training subset
        - data_test (pandas.DataFrame): The test subset

    Notes
    -----
    - If shuffle is False, the split will be performed in sequence (first n% rows
      for training, remaining for test)
    - The function preserves the index of the original DataFrame
    - For reproducible results, set both shuffle=True and specify a random_state
    """

    data_train, data_test = train_test_split(
        data,
        test_size=proportion_test_samples,
        shuffle=shuffle,
        random_state=random_state,
    )

    log.info("Train and test subsets with shuffle = " + str(shuffle))
    log.info("Train samples: " + str(data_train.shape))
    log.info("Test samples: " + str(data_test.shape))

    return data_train, data_test


def generate_features_config_default(data: pd, file_path: str, log=None) -> str:
    """
    Generates a default configuration for features based on the input dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the dataset.

    Returns
    -------
    str
        A string containing the default configuration for features.
    """
    features_config = "["
    for column in data.columns:
        features_config += f'VariableConfig(name="{column}", scaling_method=ScalingMethod.none_scaler),\n'
    features_config += "]"

    log.info("Default features configuration:")
    log.info(features_config)

    with open(file_path, "w") as file:
        file.write(features_config)

    return features_config
