"""
Xplore DS :: Data visualization with Plotly
"""

import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from pathlib import Path
import sys
import os
from sklearn.preprocessing import LabelEncoder

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


def binary_ks_curve(y_true, y_probas):
    """This function generates the points necessary to calculate the KS
    Statistic curve.

    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.

        y_probas (array-like, shape (n_samples)): Probability predictions of
            the positive class.

    Returns:
        thresholds (numpy.ndarray): An array containing the X-axis values for
            plotting the KS Statistic plot.

        pct1 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.

        pct2 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.

        ks_statistic (float): The KS Statistic, or the maximum vertical
            distance between the two curves.

        max_distance_at (float): The X-axis value at which the maximum vertical
            distance between the two curves is seen.

        classes (np.ndarray, shape (2)): An array containing the labels of the
            two classes making up `y_true`.

    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The KS Statistic
            is only relevant in binary classification.
    """
    y_true, y_probas = np.asarray(y_true), np.asarray(y_probas)
    lb = LabelEncoder()
    encoded_labels = lb.fit_transform(y_true)
    if len(lb.classes_) != 2:
        raise ValueError(
            "Cannot calculate KS statistic for data with "
            "{} category/ies".format(len(lb.classes_))
        )
    idx = encoded_labels == 0
    data1 = np.sort(y_probas[idx])
    data2 = np.sort(y_probas[np.logical_not(idx)])

    ctr1, ctr2 = 0, 0
    thresholds, pct1, pct2 = [], [], []
    while ctr1 < len(data1) or ctr2 < len(data2):

        # Check if data1 has no more elements
        if ctr1 >= len(data1):
            current = data2[ctr2]
            while ctr2 < len(data2) and current == data2[ctr2]:
                ctr2 += 1

        # Check if data2 has no more elements
        elif ctr2 >= len(data2):
            current = data1[ctr1]
            while ctr1 < len(data1) and current == data1[ctr1]:
                ctr1 += 1

        else:
            if data1[ctr1] > data2[ctr2]:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1

            elif data1[ctr1] < data2[ctr2]:
                current = data1[ctr1]
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

            else:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

        thresholds.append(current)
        pct1.append(ctr1)
        pct2.append(ctr2)

    thresholds = np.asarray(thresholds)
    pct1 = np.asarray(pct1) / float(len(data1))
    pct2 = np.asarray(pct2) / float(len(data2))

    if thresholds[0] != 0:
        thresholds = np.insert(thresholds, 0, [0.0])
        pct1 = np.insert(pct1, 0, [0.0])
        pct2 = np.insert(pct2, 0, [0.0])
    if thresholds[-1] != 1:
        thresholds = np.append(thresholds, [1.0])
        pct1 = np.append(pct1, [1.0])
        pct2 = np.append(pct2, [1.0])

    differences = pct1 - pct2
    ks_statistic, max_distance_at = (
        np.max(differences),
        thresholds[np.argmax(differences)],
    )

    return thresholds, pct1, pct2, ks_statistic, max_distance_at, lb.classes_


def plot_ks_statistic(
    y_true,
    y_probas,
    title="KS Statistic Plot",
    ax=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
):
    """Generates the KS Statistic plot from labels and scores/probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "KS Statistic Plot".

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> lr = LogisticRegression()
        >>> lr = lr.fit(X_train, y_train)
        >>> y_probas = lr.predict_proba(X_test)
        >>> skplt.metrics.plot_ks_statistic(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_ks_statistic.png
           :align: center
           :alt: KS Statistic
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(
            "Cannot calculate KS statistic for data with "
            "{} category/ies".format(len(classes))
        )
    probas = y_probas

    # Compute KS Statistic curves
    thresholds, pct1, pct2, ks_statistic, max_distance_at, classes = binary_ks_curve(
        y_true, probas[:, 1].ravel()
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot(thresholds, pct1, lw=3, label="Class {}".format(classes[0]))
    ax.plot(thresholds, pct2, lw=3, label="Class {}".format(classes[1]))
    idx = np.where(thresholds == max_distance_at)[0][0]
    ax.axvline(
        max_distance_at,
        *sorted([pct1[idx], pct2[idx]]),
        label="KS Statistic: {:.3f} at {:.3f}".format(ks_statistic, max_distance_at),
        linestyle=":",
        lw=3,
        color="black"
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.set_xlabel("Threshold", fontsize=text_fontsize)
    ax.set_ylabel("Percentage below threshold", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc="lower right", fontsize=text_fontsize)

    return ax
