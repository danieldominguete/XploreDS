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

from xplore_ds.data_visualization.data_viz_plotly import plot_scatter_2d


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


def get_evaluation_regression_metrics(y_true, y_pred, results_folder, log):

    log.info(
        "Mean Absolute Error: {a:.3f}".format(
            a=get_mean_absolute_error(y_true=y_true, y_pred=y_pred)
        )
    )
    log.info(
        "Median Absolute Error: {a:.3f}".format(
            a=get_mdae_error_score(y_true=y_true, y_pred=y_pred)
        )
    )
    log.info(
        "Mean Squared Error: {a:.3f}".format(
            a=get_mse_error_score(y_true=y_true, y_pred=y_pred)
        )
    )
    log.info("R2: {a:.3f}".format(a=get_r2_score(y_true=y_true, y_pred=y_pred)))
    log.info(
        "Explained Variance: {a:.3f}".format(
            a=get_explained_variance_score(y_true=y_true, y_pred=y_pred)
        )
    )
    log.info(
        "Max Absolute Error: {a:.3f}".format(
            a=get_max_error_score(y_true=y_true, y_pred=y_pred)
        )
    )


def plot_evaluation_regression_results(
    y_true, y_pred, results_folder, view_charts, save_charts, log
):

    # scatter predicao x target
    log.info("Plotting scatter plot of target x predicted")
    file_path = results_folder + "scatter_pred_x_target.png"
    plot_scatter_2d(
        x=y_true,
        y=y_pred,
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

    y_true = data[y_target_column_name]
    y_pred = data[y_predict_column_name]

    get_evaluation_regression_metrics(
        y_true=y_true, y_pred=y_pred, results_folder=results_folder, log=log
    )
    plot_evaluation_regression_results(
        y_true=y_true,
        y_pred=y_pred,
        results_folder=results_folder,
        view_charts=view_charts,
        save_charts=save_charts,
        log=log,
    )
