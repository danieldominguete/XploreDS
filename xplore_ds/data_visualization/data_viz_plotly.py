"""
Xplore DS :: Data visualization with Plotly
"""

import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path
import sys
import os

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xplore_ds.data_handler.file import create_folder


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


def plot_confusion_matrix(confusion_matrix, class_names):

    confusion_matrix = confusion_matrix.astype(int)

    layout = {
        "title": "Confusion Matrix",
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
    }

    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix, x=class_names, y=class_names, hoverongaps=False
        ),
        layout=layout,
    )
    fig.show()


def plot_confusion_matrix(
    confusion_matrix: object,
    labels: list,
    title: str = "",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):
    # create the heatmap
    heatmap = go.Heatmap(z=confusion_matrix, x=labels, y=labels, colorscale="Viridis")

    # create the layout
    layout = go.Layout(title=title)

    # create the figure
    fig = go.Figure(data=[heatmap], layout=layout)

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)
