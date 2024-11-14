"""
Xplore DS :: Data visualization with Plotly
"""

import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from pathlib import Path
import sys
import os

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xploreds.data_handler.file import create_folder


def deploy_chart_in_navigator(fig: object) -> None:
    fig.show()


def save_chart_file(fig: object, path: str = None) -> None:
    create_folder(os.path.dirname(path))
    fig.write_image(path)


# https://plotly.com/python-api-reference/generated/plotly.express.scatter.html
def plot_scatter_2d(
    data,
    x_col_name,
    y_col_name,
    title: str = "",
    marginal_plot: bool = True,
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):
    if marginal_plot:
        fig = px.scatter(
            data_frame=data,
            x=x_col_name,
            y=y_col_name,
            title=title,
            marginal_x="histogram",
            marginal_y="histogram",
        )
    else:
        fig = px.scatter(data_frame=data, x=x_col_name, y=y_col_name, title=title)

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)


def plot_histogram(
    data,
    x_col_name,
    title: str = "",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):
    fig = px.histogram(data_frame=data, x=x_col_name, title=title)

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)


def plot_confusion_matrix(
    confusion_matrix: object,
    x_y_labels: list,
    title: str = "",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):

    fig = px.imshow(
        confusion_matrix,
        text_auto="0.2f",
        aspect="auto",
        title=title,
        x=x_y_labels,
        y=x_y_labels,
    )
    fig.update_xaxes(visible=True, title_text="Predicted value")
    fig.update_yaxes(visible=True, title_text="Real value")

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)


def plot_precision_recall_curve(
    precision: list,
    recall: list,
    title: str = "",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):

    fig = px.area(
        x=recall,
        y=precision,
        title=title,
    )
    fig.update_xaxes(title_text="Recall")
    fig.update_yaxes(title_text="Precision")

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)


def plot_roc_curve(
    fpr: list,
    tpr: list,
    title: str = "",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):

    fig = px.area(
        x=fpr,
        y=tpr,
        title=title,
    )
    fig.update_xaxes(title_text="False Positive Rate")
    fig.update_yaxes(title_text="True Positive Rate")

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)
