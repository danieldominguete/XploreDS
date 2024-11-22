"""
Xplore DS :: Data visualization with Plotly
"""

import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xploreds.data_handler.file import create_folder
from xploreds.data_analysis.statistics import get_binary_ks_curve
from xploreds.data_analysis.statistics import get_ks_score_over_time


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
    precision_no_skill: list = None,
    recall_no_skill: list = None,
    title: str = "",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):

    # plot com referencia de classificador baseline
    if precision_no_skill is not None and recall_no_skill is not None:
        fig = px.area(
            x=recall,
            y=precision,
            title=title,
        )
        fig.add_scatter(
            x=recall_no_skill, y=precision_no_skill, mode="lines", name="no skill model"
        )
        fig.update_xaxes(title_text="Recall")
        fig.update_yaxes(title_text="Precision")
    else:
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
    fpr_no_skill: list = None,
    tpr_no_skill: list = None,
    title: str = "",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):

    # plot com referencia de classificador baseline
    if fpr_no_skill is not None and tpr_no_skill is not None:
        fig = px.area(
            x=fpr,
            y=tpr,
            title=title,
        )
        fig.add_scatter(
            x=fpr_no_skill, y=tpr_no_skill, mode="lines", name="no skill model"
        )
        fig.update_xaxes(title_text="False Positive Rate")
        fig.update_yaxes(title_text="True Positive Rate")
    else:
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
    text_col_name: str = None,
    barmode: str = "group",
    orientation: str = "v",
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
        data_frame=data,
        x=x_col_name,
        y=y_col_name,
        text=text_col_name,
        barmode=barmode,
        orientation=orientation,
        color=color_col_name,
        title=title,
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


def plot_scatter_plus_line(
    data,
    x_line_col_name: str,
    y_line_col_name: str,
    x_scatter_col_name: str,
    y_scatter_col_name: str,
    z_scatter_size_col_name: str = None,
    x_range_limits: list = None,
    y_range_limits: list = None,
    group_color_col_name: str = None,
    title: str = "",
    axis_names: list = None,
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):

    fig = px.scatter(
        data_frame=data,
        x=x_scatter_col_name,
        y=y_scatter_col_name,
        size=z_scatter_size_col_name,
        color=group_color_col_name,
        symbol=group_color_col_name,
        title=title,
        labels=axis_names,
    )

    fig.add_trace(
        px.line(
            data_frame=data,
            x=x_line_col_name,
            y=y_line_col_name,
            color_discrete_sequence=["black"],
        ).data
    )

    if x_range_limits is not None:
        fig.update_xaxes(range=x_range_limits)

    if y_range_limits is not None:
        fig.update_yaxes(range=y_range_limits)

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)


def plot_ks_statistic(
    data: pd,
    y_true_column_name: str,
    y_probas_column_name: str,
    title="KS Statistic Plot",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):

    # Compute KS Statistic curves
    thresholds, pct1, pct2, ks_statistic, max_distance_at, classes = (
        get_binary_ks_curve(
            data=data,
            y_true_column_name=y_true_column_name,
            y_probas_column_name=y_probas_column_name,
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=pct1,
            mode="lines",
            name="Class {}".format(classes[0]),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=pct2,
            mode="lines",
            name="Class {}".format(classes[1]),
        )
    )

    idx = np.where(thresholds == max_distance_at)[0][0]
    ks_x = [max_distance_at, max_distance_at]
    ks_y = [pct1[idx], pct2[idx]]
    fig.add_trace(
        go.Scatter(
            x=ks_x,
            y=ks_y,
            mode="lines",
            name="KS Statistic: {:.3f} at {:.3f}".format(ks_statistic, max_distance_at),
            line=dict(color="black", dash="dash"),
        )
    )

    fig.update_layout(
        xaxis_title="Threshold",
        yaxis_title="Percentage below threshold",
        legend_title="Legend",
        title=title,
    )

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)


def plot_histogram_binary_classes(
    data,
    x_col_name: str,
    y_target_col_name: str,
    cut_offs: list = None,
    title: str = "",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):

    fig = px.histogram(
        data_frame=data,
        x=x_col_name,
        color=y_target_col_name,
        facet_col=y_target_col_name,
        barmode="overlay",
        marginal="box",
        title=title,
    )

    if cut_offs is not None:
        for cut_off in cut_offs:
            fig.add_vline(x=cut_off, line_width=3, line_dash="dash", line_color="green")

    fig.update_layout(
        xaxis_title="Threshold",
        yaxis_title="Frequency",
        legend_title="Legend",
        title=title,
    )

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)


def plot_distribution_binary_classes(
    data,
    x_col_name: str,
    y_target_col_name: str,
    cut_offs: list = None,
    title: str = "",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):

    hist1 = data[x_col_name][data[y_target_col_name] == 0]
    hist2 = data[x_col_name][data[y_target_col_name] == 1]
    hist_data = [hist1, hist2]
    group_labels = ["0", "1"]

    fig = ff.create_distplot(
        hist_data=hist_data,
        group_labels=group_labels,
        bin_size=0.025,
        show_curve=True,
        show_hist=False,
        show_rug=False,
    )

    if cut_offs is not None:
        for cut_off in cut_offs:
            fig.add_vline(x=cut_off, line_width=3, line_dash="dash", line_color="green")

    fig.update_layout(
        xaxis_title="Threshold",
        yaxis_title="Frequency",
        legend_title="Legend",
        title=title,
    )

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)


def plot_ks_score_over_time(
    data: pd,
    y_true_column_name: str,
    y_probas_column_name: str,
    date_column_name: str,
    title="KS Statistic over time",
    view_chart: bool = True,
    save_chart: bool = False,
    file_path_image: str = None,
):

    time_frame, ks_values, ks_ci_low_values, ks_ci_high_values = get_ks_score_over_time(
        data=data,
        y_true_column_name=y_true_column_name,
        y_probas_column_name=y_probas_column_name,
        time_column_name=date_column_name,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time_frame,
            y=ks_ci_low_values,
            mode="lines",
            name="CI low",
            line=dict(color="lightblue", dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=time_frame,
            y=ks_ci_high_values,
            mode="lines",
            name="CI high",
            fill="tonexty",
            fillcolor="lightblue",
            line=dict(color="lightblue", dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=time_frame,
            y=ks_values,
            mode="lines+markers",
            name="KS Statistics",
            line=dict(color="blue"),
            marker=dict(symbol="circle", size=8, color="blue"),
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="KS Statistic",
        legend_title="Legend",
        title=title,
        yaxis=dict(range=[0, 1]),
    )

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)
