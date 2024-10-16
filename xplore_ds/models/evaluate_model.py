"""
Xplore DS :: Evaluate models
"""

from pathlib import Path
import sys, os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xplore_ds.data_visualization.data_viz_plotly import (
    plot_scatter_2d,
    plot_histogram,
    plot_confusion_matrix,
)
from sklearn.metrics import confusion_matrix


def get_mean_absolute_error(y_true, y_pred):
    return mean_absolute_error(y_true=y_true, y_pred=y_pred)


def get_explained_variance_score(y_true, y_pred):
    return explained_variance_score(y_true=y_true, y_pred=y_pred)


def get_max_error_score(y_true, y_pred):
    return max_error(y_true=y_true, y_pred=y_pred)


def get_mse_error_score(y_true, y_pred):
    return mean_squared_error(y_true=y_true, y_pred=y_pred)


def get_mdae_error_score(y_true, y_pred):
    return median_absolute_error(y_true=y_true, y_pred=y_pred)


def get_r2_score(y_true, y_pred):
    return r2_score(y_true=y_true, y_pred=y_pred)


def get_accuracy_score(y_true, y_pred):
    return accuracy_score(y_true=y_true, y_pred=y_pred)


def get_balanced_accuracy_score(y_true, y_pred):
    return balanced_accuracy_score(y_true=y_true, y_pred=y_pred)


def get_confusion_matrix(y_true_labels, y_pred_labels, labels):

    cm = confusion_matrix(
        y_true=y_true_labels, y_pred=y_pred_labels, labels=labels, normalize="all"
    )
    return cm


def get_evaluation_regression_metrics(
    data, y_target_col_name, y_predict_col_name, results_folder, log
):

    log.info(
        "Mean Absolute Error: {a:.3f}".format(
            a=get_mean_absolute_error(
                y_true=data[y_target_col_name], y_pred=data[y_predict_col_name]
            )
        )
    )
    log.info(
        "Median Absolute Error: {a:.3f}".format(
            a=get_mdae_error_score(
                y_true=data[y_target_col_name], y_pred=data[y_predict_col_name]
            )
        )
    )
    log.info(
        "Mean Squared Error: {a:.3f}".format(
            a=get_mse_error_score(
                y_true=data[y_target_col_name], y_pred=data[y_predict_col_name]
            )
        )
    )
    log.info(
        "R2: {a:.3f}".format(
            a=get_r2_score(
                y_true=data[y_target_col_name], y_pred=data[y_predict_col_name]
            )
        )
    )
    log.info(
        "Explained Variance: {a:.3f}".format(
            a=get_explained_variance_score(
                y_true=data[y_target_col_name], y_pred=data[y_predict_col_name]
            )
        )
    )
    log.info(
        "Max Absolute Error: {a:.3f}".format(
            a=get_max_error_score(
                y_true=data[y_target_col_name], y_pred=data[y_predict_col_name]
            )
        )
    )


def get_evaluation_binary_classification_metrics(
    data, y_target_col_name, y_predict_col_name, results_folder, log
):

    log.info(
        "Mean Absolute Error: {a:.3f}".format(
            a=get_mean_absolute_error(
                y_true=data[y_target_col_name], y_pred=data[y_predict_col_name]
            )
        )
    )
    log.info(
        "Median Absolute Error: {a:.3f}".format(
            a=get_mdae_error_score(
                y_true=data[y_target_col_name], y_pred=data[y_predict_col_name]
            )
        )
    )
    log.info(
        "Mean Squared Error: {a:.3f}".format(
            a=get_mse_error_score(
                y_true=data[y_target_col_name], y_pred=data[y_predict_col_name]
            )
        )
    )
    log.info(
        "R2: {a:.3f}".format(
            a=get_r2_score(
                y_true=data[y_target_col_name], y_pred=data[y_predict_col_name]
            )
        )
    )
    log.info(
        "Explained Variance: {a:.3f}".format(
            a=get_explained_variance_score(
                y_true=data[y_target_col_name], y_pred=data[y_predict_col_name]
            )
        )
    )
    log.info(
        "Max Absolute Error: {a:.3f}".format(
            a=get_max_error_score(
                y_true=data[y_target_col_name], y_pred=data[y_predict_col_name]
            )
        )
    )


def plot_evaluation_regression_results(
    data,
    y_target_col_name,
    y_predict_col_name,
    results_folder,
    view_charts,
    save_charts,
    log,
):

    # scatter predicao x target
    log.info("Plotting scatter plot of target x predicted")
    file_path = results_folder + "scatter_pred_x_target.png"

    plot_scatter_2d(
        data=data,
        x_col_name=y_target_col_name,
        y_col_name=y_predict_col_name,
        title="Scatter plot of target x predicted",
        marginal_plot=True,
        view_chart=view_charts,
        save_chart=save_charts,
        file_path_image=file_path,
    )

    # distribuicao dos residuos
    log.info("Plotting histogram of residuals")
    file_path = results_folder + "histogram_residuals.png"

    data["_residuo"] = data[y_target_col_name] - data[y_predict_col_name]

    plot_histogram(
        data=data,
        x_col_name="_residuo",
        title="Histogram of residuals",
        view_chart=view_charts,
        save_chart=save_charts,
        file_path_image=file_path,
    )


def plot_evaluation_binary_classification_results(
    data,
    y_target_col_name,
    y_target_label_col_name,
    y_predict_col_name,
    y_predict_label_col_name,
    labels,
    results_folder,
    view_charts,
    save_charts,
    log,
):

    # scatter predicao x target
    log.info("Plotting scatter plot of target x predicted")
    file_path = results_folder + "scatter_pred_x_target.png"

    plot_scatter_2d(
        data=data,
        x_col_name=y_target_col_name,
        y_col_name=y_predict_col_name,
        title="Scatter plot of target x predicted",
        marginal_plot=True,
        view_chart=view_charts,
        save_chart=save_charts,
        file_path_image=file_path,
    )

    # confusion matrix
    log.info("Plotting confusion matrix")
    file_path = results_folder + "confusion_matrix.png"

    cm = get_confusion_matrix(
        y_true_labels=data[y_target_label_col_name],
        y_pred_labels=data[y_predict_label_col_name],
        labels=labels,
    )

    plot_confusion_matrix(
        confusion_matrix=cm,
        labels=labels,
        title="Confusion Matrix",
        view_chart=view_charts,
        save_chart=save_charts,
        file_path_image=file_path,
    )


def evaluate_regression(
    data: pd,
    y_predict_column_name: str,
    y_target_column_name: str,
    results_folder: str = None,
    view_charts: bool = True,
    save_charts: bool = True,
    log: object = None,
):

    log.info(
        "=================================================================================="
    )
    log.info("Evaluating regression model metrics...")

    results_folder_metrics = results_folder + "metrics/"
    get_evaluation_regression_metrics(
        data,
        y_target_col_name=y_target_column_name,
        y_predict_col_name=y_predict_column_name,
        results_folder=results_folder_metrics,
        log=log,
    )

    log.info(
        "=================================================================================="
    )
    log.info("Evaluating regression model data visualization...")
    results_folder_charts = results_folder + "charts/"
    plot_evaluation_regression_results(
        data=data,
        y_target_col_name=y_target_column_name,
        y_predict_col_name=y_predict_column_name,
        results_folder=results_folder_charts,
        view_charts=view_charts,
        save_charts=save_charts,
        log=log,
    )


def evaluate_binary_classification(
    data: pd,
    y_predict_column_name: str,
    y_target_column_name: str,
    y_predict_class_column_name: str,
    y_target_class_column_name: str,
    labels: list,
    results_folder: str = None,
    view_charts: bool = True,
    save_charts: bool = True,
    log: object = None,
):

    log.info(
        "=================================================================================="
    )
    log.info("Evaluating binary classification model metrics...")

    results_folder_metrics = results_folder + "metrics/"

    get_evaluation_binary_classification_metrics(
        data,
        y_target_col_name=y_target_column_name,
        y_predict_col_name=y_predict_column_name,
        results_folder=results_folder_metrics,
        log=log,
    )

    log.info(
        "=================================================================================="
    )
    log.info("Evaluating binary classification model data visualization...")

    results_folder_charts = results_folder + "charts/"

    plot_evaluation_binary_classification_results(
        data=data,
        y_target_col_name=y_target_column_name,
        y_target_label_col_name=y_target_class_column_name,
        y_predict_col_name=y_predict_column_name,
        y_predict_label_col_name=y_predict_class_column_name,
        labels=labels,
        results_folder=results_folder_charts,
        view_charts=view_charts,
        save_charts=save_charts,
        log=log,
    )
