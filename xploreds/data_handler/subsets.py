"""
Xplore DS :: Dataset Tools Package
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xploreds.data_analysis.drift import calculate_psi_score
from xploreds.data_visualization.data_viz_plotly import plot_heatmap_simple, plot_bar


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


def check_drift_subsets(
    data_train=None,
    data_oos=None,
    data_oot=None,
    log=None,
    view_plots: bool = True,
    save_plots: bool = True,
    output_folder_path: str = None,
    prefix_label: str = None,
) -> pd:

    # PSI score
    variables_psi_analysis = pd.DataFrame(
        index=data_train.columns,
        columns=["train_oos", "train_oot"],
    )

    # Get all numeric columns including int, float and their variants
    numerical_variables = data_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_variables = data_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if data_oos is not None and data_oos.shape[0] > 0:
        if len(numerical_variables) > 0:

            for var in numerical_variables:

                value, drift_detected = calculate_psi_score(
                    reference_data=data_train,
                    current_data=data_oos,
                    column_name=var,
                    feature_type="num",
                    threshold=0.2,
                )

                variables_psi_analysis["train_oos"].loc[var] = value

                if drift_detected:
                    log.warning(
                        "PSI Drift detected: "
                        + str(var)
                        + " at "
                        + str("train_oos")
                        + " = {:.4f}".format(value)
                    )
                else:
                    log.info(
                        "PSI: "
                        + str(var)
                        + " at "
                        + str("train_oos")
                        + " = {:.4f}".format(value)
                    )

            if len(categorical_variables) > 0:

                for var in categorical_variables:

                    value, drift_detected = calculate_psi_score(
                        reference_data=data_train,
                        current_data=data_oos,
                        column_name=var,
                        feature_type="cat",
                        threshold=0.2,
                    )

                    variables_psi_analysis["train_oos"].loc[var] = value

                    if drift_detected:
                        log.warning(
                            "PSI Drift detected: "
                            + str(var)
                            + " at "
                            + str("train_oos")
                            + " = {:.4f}".format(value)
                        )
                    else:
                        log.info(
                            "PSI: "
                            + str(var)
                            + " at "
                            + str("train_oos")
                            + " = {:.4f}".format(value)
                        )
    if data_oot is not None and data_oot.shape[0] > 0:
        if len(numerical_variables) > 0:

            for var in numerical_variables:
                value, drift_detected = calculate_psi_score(
                    reference_data=data_train,
                    current_data=data_oot,
                    column_name=var,
                    feature_type="num",
                    threshold=0.2,
                )

                variables_psi_analysis["train_oot"].loc[var] = value

                if drift_detected:
                    log.warning(
                        "PSI Drift detected: "
                        + str(var)
                        + " at "
                        + str("train_oot")
                        + " = {:.4f}".format(value)
                    )
                else:
                    log.info(
                        "PSI: "
                        + str(var)
                        + " at "
                        + str("train_oot")
                        + " = {:.4f}".format(value)
                    )

        if len(categorical_variables) > 0:

            for var in categorical_variables:

                value, drift_detected = calculate_psi_score(
                    reference_data=data_train,
                    current_data=data_oot,
                    column_name=var,
                    feature_type="cat",
                    threshold=0.2,
                )

                variables_psi_analysis["train_oot"].loc[var] = value

                if drift_detected:
                    log.warning(
                        "PSI Drift detected: "
                        + str(var)
                        + " at "
                        + str("train_oot")
                        + " = {:.4f}".format(value)
                    )
                else:
                    log.info(
                        "PSI: "
                        + str(var)
                        + " at "
                        + str("train_oot")
                        + " = {:.4f}".format(value)
                    )

    # Drift only for aggregate datetime
    if log:
        log.info("Plotting PSI drift analysis...")

    variables_psi_analysis = variables_psi_analysis.reset_index()
    variables_psi_analysis = variables_psi_analysis.rename(
        columns={"index": "variable"}
    )

    variables_psi_analysis = variables_psi_analysis.sort_values(
        by=["train_oos"], ascending=True
    )

    plot_bar(
        data=variables_psi_analysis,
        y_col_name="variable",
        x_col_name="train_oos",
        orientation="h",
        title="PSI subsets drift analysis - Train x Out of Sample",
        view_chart=view_plots,
        save_chart=save_plots,
        file_path_image=output_folder_path
        + "/charts/"
        + prefix_label
        + "_oos_psi_drift_analysis.png",
    )

    variables_psi_analysis = variables_psi_analysis.sort_values(
        by=["train_oot"], ascending=True
    )

    plot_bar(
        data=variables_psi_analysis,
        y_col_name="variable",
        x_col_name="train_oot",
        orientation="h",
        title="PSI subsets drift analysis - Train x Out of Time",
        view_chart=view_plots,
        save_chart=save_plots,
        file_path_image=output_folder_path
        + "/charts/"
        + prefix_label
        + "_oot_psi_drift_analysis.png",
    )

    # plot_heatmap_simple(
    #     data=variables_psi_analysis,
    #     x_ref_col_name="variable",
    #     y_values_col_list=["train_oos", "train_oot"],
    #     title="PSI subsets drift analysis",
    #     view_chart=view_plots,
    #     save_chart=save_plots,
    #     show_values=False,
    #     file_path_image=output_folder_path
    #     + "/charts/"
    #     + prefix_label
    #     + "_psi_drift_analysis.png",
    # )
