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
    """
    Display a Plotly figure in the default web browser.

    Args:
        fig (object): Plotly figure object to display

    Returns:
        None
    """

    fig.show()


def save_chart_file(fig: object, path: str = None) -> None:
    """
    Save a Plotly figure as an image file.

    Args:
        fig (object): Plotly figure object to save
        path (str, optional): File path where to save the image. Defaults to None.

    Returns:
        None
    """

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
    """
    Create a 2D scatter plot using Plotly Express.

    Args:
        data: pandas DataFrame containing the data to plot
        x_col_name: Name of the column to plot on x-axis
        y_col_name: Name of the column to plot on y-axis
        title (str, optional): Title of the plot. Defaults to "".
        marginal_plot (bool, optional): Whether to include marginal histograms. Defaults to True.
        view_chart (bool, optional): Whether to display the chart in browser. Defaults to True.
        save_chart (bool, optional): Whether to save the chart as image. Defaults to False.
        file_path_image (str, optional): Path where to save the image. Required if save_chart is True.

    Returns:
        None
    """

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
    """
    Create a histogram plot using Plotly Express.

    Args:
        data: pandas DataFrame containing the data to plot
        x_col_name: Name of the column to create histogram from
        title (str, optional): Title of the plot. Defaults to "".
        view_chart (bool, optional): Whether to display the chart in browser. Defaults to True.
        save_chart (bool, optional): Whether to save the chart as image. Defaults to False.
        file_path_image (str, optional): Path where to save the image. Required if save_chart is True.

    Returns:
        None
    """

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
    """
    Create a confusion matrix visualization using Plotly Express.

    Args:
        confusion_matrix (object): The confusion matrix to visualize
        x_y_labels (list): Labels for both x and y axes
        title (str, optional): Title of the plot. Defaults to "".
        view_chart (bool, optional): Whether to display the chart in browser. Defaults to True.
        save_chart (bool, optional): Whether to save the chart as image. Defaults to False.
        file_path_image (str, optional): Path where to save the image. Required if save_chart is True.

    Returns:
        None
    """

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
    """
    Create a precision-recall curve plot using Plotly Express.

    Args:
        precision (list): List of precision values
        recall (list): List of recall values
        title (str, optional): Title of the plot. Defaults to "".
        view_chart (bool, optional): Whether to display the chart in browser. Defaults to True.
        save_chart (bool, optional): Whether to save the chart as image. Defaults to False.
        file_path_image (str, optional): Path where to save the image. Required if save_chart is True.

    Returns:
        None
    """

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
    """
    Create a ROC curve plot using Plotly Express.

    Args:
        fpr (list): List of false positive rates
        tpr (list): List of true positive rates
        title (str, optional): Title of the plot. Defaults to "".
        view_chart (bool, optional): Whether to display the chart in browser. Defaults to True.
        save_chart (bool, optional): Whether to save the chart as image. Defaults to False.
        file_path_image (str, optional): Path where to save the image. Required if save_chart is True.

    Returns:
        None
    """
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


def plot_bar(
    data,
    x_col_name: str,
    y_col_name: str,
    color_col_name: str = None,
    title: str = "",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):
    """
    Creates a bar chart visualization using Plotly Express.

    Args:
        data (pd.DataFrame): Input DataFrame containing the data to plot.
        x_col_name (str): Column name for x-axis categories.
        y_col_name (str): Column name for y-axis values.
        color_col_name (str, optional): Column name for color grouping. Defaults to None.
        title (str, optional): Title of the plot. Defaults to empty string.
        view_chart (bool, optional): If True, displays the chart in browser. Defaults to True.
        save_chart (bool, optional): If True, saves the chart as image. Defaults to False.
        file_path_image (str, optional): Path to save the image file. Required if save_chart is True.

    Returns:
        None

    """

    fig = px.bar(
        data_frame=data, x=x_col_name, y=y_col_name, color=color_col_name, title=title
    )

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)


def plot_lines(
    data,
    x_col_name: str,
    y_col_name: str,
    color_col_name: str = None,
    title: str = "",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):
    """
    Creates a line chart visualization using Plotly Express.

    Args:
        data (pd.DataFrame): Input DataFrame containing the data to plot.
        x_col_name (str): Column name for x-axis values.
        y_col_name (str): Column name for y-axis values.
        color_col_name (str, optional): Column name for color grouping. Defaults to None.
        title (str, optional): Title of the plot. Defaults to empty string.
        view_chart (bool, optional): If True, displays the chart in browser. Defaults to True.
        save_chart (bool, optional): If True, saves the chart as image. Defaults to False.
        file_path_image (str, optional): Path to save the image file. Required if save_chart is True.

    Returns:
        None

    """

    fig = px.line(
        data_frame=data, x=x_col_name, y=y_col_name, color=color_col_name, title=title
    )

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)


def plot_boxplot(
    data,
    x_col_name: str,
    y_col_name: str,
    title: str = "",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):

    fig = px.box(data_frame=data, x=x_col_name, y=y_col_name, title=title)

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)
