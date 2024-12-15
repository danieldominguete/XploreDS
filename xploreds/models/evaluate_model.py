"""
Xplore DS :: Evaluate models
"""

from pathlib import Path
import sys, os

import numpy as np
import pandas as pd

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xploreds.data_visualization.data_viz_plotly import (
    plot_scatter_2d,
    plot_histogram,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_ks_statistic,
    plot_histogram_binary_classes,
    plot_distribution_binary_classes,
    plot_ks_score_over_time,
    plot_pr_auc_score_over_time,
    plot_roc_auc_score_over_time,
)
from xploreds.data_handler.file import save_dictionary_to_json
from xploreds.data_analysis.statistics import (
    get_ks_score_for_binary_classifier,
    get_ks_score_confidence_interval_for_binary_classifier,
    get_accuracy_score,
    get_balanced_accuracy_score,
    get_binary_ks_curve,
    get_classification_report,
    get_confusion_matrix,
    get_explained_variance_score,
    get_gini_score_for_binary_classifier,
    get_ks_score_over_time,
    get_ks_statistics_interval_confidence,
    get_max_error_score,
    get_mdae_error_score,
    get_mean_absolute_error,
    get_mse_error_score,
    get_precision_recall_score_for_binary_classifier,
    get_pr_auc_statistics_interval_confidence,
    get_r2_score,
    get_roc_auc_score_for_binary_classifier,
    get_roc_auc_statistics_interval_confidence,
    precision_recall_curve,
    roc_curve,
)


def get_evaluation_regression_metrics(
    data, y_target_col_name, y_predict_col_name, results_folder, log
):

    log.info(
        "Mean Absolute Error: {a:.3f}".format(
            a=get_mean_absolute_error(
                y_numerical_true=data[y_target_col_name],
                y_numerical_pred=data[y_predict_col_name],
            )
        )
    )
    log.info(
        "Median Absolute Error: {a:.3f}".format(
            a=get_mdae_error_score(
                y_numerical_true=data[y_target_col_name],
                y_numerical_pred=data[y_predict_col_name],
            )
        )
    )
    log.info(
        "Mean Squared Error: {a:.3f}".format(
            a=get_mse_error_score(
                y_numerical_true=data[y_target_col_name],
                y_numerical_pred=data[y_predict_col_name],
            )
        )
    )
    log.info(
        "R2: {a:.3f}".format(
            a=get_r2_score(
                y_numerical_true=data[y_target_col_name],
                y_numerical_pred=data[y_predict_col_name],
            )
        )
    )
    log.info(
        "Explained Variance: {a:.3f}".format(
            a=get_explained_variance_score(
                y_numerical_true=data[y_target_col_name],
                y_numerical_pred=data[y_predict_col_name],
            )
        )
    )
    log.info(
        "Max Absolute Error: {a:.3f}".format(
            a=get_max_error_score(
                y_numerical_true=data[y_target_col_name],
                y_numerical_pred=data[y_predict_col_name],
            )
        )
    )


def get_common_evaluation_binary_classification_metrics(
    data,
    y_target_numerical_col_name: str,
    y_predict_numerical_col_name: str,
    y_no_skill_predict_numerical_col_name: str = None,
    log=None,
):

    metrics = {}

    # mean absolute error
    v = get_mean_absolute_error(
        y_numerical_true=data[y_target_numerical_col_name],
        y_numerical_pred=data[y_predict_numerical_col_name],
    )
    log.info("Mean Absolute Error: {a:.3f}".format(a=v))
    metrics["Mean Absolute Error"] = v

    if y_no_skill_predict_numerical_col_name:
        v = get_mean_absolute_error(
            y_numerical_true=data[y_target_numerical_col_name],
            y_numerical_pred=data[y_no_skill_predict_numerical_col_name],
        )
        log.info("Mean Absolute Error (No Skill): {a:.3f}".format(a=v))
        metrics["Mean Absolute Error (No Skill)"] = v

    v = get_mse_error_score(
        y_numerical_true=data[y_target_numerical_col_name],
        y_numerical_pred=data[y_predict_numerical_col_name],
    )
    log.info("Mean Squared Error: {a:.3f}".format(a=v))
    metrics["Mean Squared Error"] = v

    # MSE
    if y_no_skill_predict_numerical_col_name:
        v = get_mse_error_score(
            y_numerical_true=data[y_target_numerical_col_name],
            y_numerical_pred=data[y_no_skill_predict_numerical_col_name],
        )
        log.info("Mean Squared Error (No Skill): {a:.3f}".format(a=v))
        metrics["Mean Absolute Error (No Skill)"] = v

    # ROC AUC
    v = get_roc_auc_score_for_binary_classifier(
        y_numerical_true=data[y_target_numerical_col_name],
        y_numerical_score_pred=data[y_predict_numerical_col_name],
    )
    log.info("ROC AUC Score: {a:.3f}".format(a=v))
    metrics["ROC AUC Score"] = v

    if y_no_skill_predict_numerical_col_name:
        v = get_roc_auc_score_for_binary_classifier(
            y_numerical_true=data[y_target_numerical_col_name],
            y_numerical_score_pred=data[y_no_skill_predict_numerical_col_name],
        )
        log.info("ROC AUC Score (No Skill): {a:.3f}".format(a=v))
        metrics["ROC AUC Score (No Skill)"] = v

    c_lower, c_upper, v = get_roc_auc_statistics_interval_confidence(
        y_numerical_true=data[y_target_numerical_col_name],
        y_numerical_score_pred=data[y_predict_numerical_col_name],
    )
    log.info(
        "ROC AUC Score Confidence Interval: {a:.3f} [{b:.3f} , {c:.3f}]".format(
            a=v, b=c_lower, c=c_upper
        )
    )

    # PRECISION RECALL
    v = get_precision_recall_score_for_binary_classifier(
        y_numerical_true=data[y_target_numerical_col_name],
        y_numerical_score_pred=data[y_predict_numerical_col_name],
    )
    log.info("Precision Recall AUC Score: {a:.3f}".format(a=v))
    metrics["Precision Recall AUC Score"] = v

    if y_no_skill_predict_numerical_col_name:
        v = get_precision_recall_score_for_binary_classifier(
            y_numerical_true=data[y_target_numerical_col_name],
            y_numerical_score_pred=data[y_no_skill_predict_numerical_col_name],
        )
        log.info("Precision Recall AUC Score (No Skill): {a:.3f}".format(a=v))
        metrics["Precision Recall AUC Score (No Skill)"] = v

    # intervalo de confianaca
    c_lower, c_upper, v = get_pr_auc_statistics_interval_confidence(
        y_numerical_true=data[y_target_numerical_col_name],
        y_numerical_score_pred=data[y_predict_numerical_col_name],
    )
    log.info(
        "PR AUC Score Confidence Interval: {a:.3f} [{b:.3f} , {c:.3f}]".format(
            a=v, b=c_lower, c=c_upper
        )
    )

    # GINI
    v = get_gini_score_for_binary_classifier(
        y_numerical_true=data[y_target_numerical_col_name],
        y_numerical_score_pred=data[y_predict_numerical_col_name],
    )
    log.info("Gini Score: {a:.3f}".format(a=v))
    metrics["Gini Score"] = v

    if y_no_skill_predict_numerical_col_name:
        v = get_gini_score_for_binary_classifier(
            y_numerical_true=data[y_target_numerical_col_name],
            y_numerical_score_pred=data[y_no_skill_predict_numerical_col_name],
        )
        log.info("Gini Score (No Skill): {a:.3f}".format(a=v))
        metrics["Gini Score (No Skill)"] = v

    # KS SCORE
    v = get_ks_score_for_binary_classifier(
        y_numerical_true=data[y_target_numerical_col_name],
        y_numerical_score_pred=data[y_predict_numerical_col_name],
    )
    log.info("KS Score: {a:.3f}".format(a=v))
    metrics["KS Score"] = v

    if y_no_skill_predict_numerical_col_name:
        v = get_ks_score_for_binary_classifier(
            y_numerical_true=data[y_target_numerical_col_name],
            y_numerical_score_pred=data[y_no_skill_predict_numerical_col_name],
        )
        log.info("KS Score (No Skill): {a:.3f}".format(a=v))
        metrics["KS Score (No Skill)"] = v

    ci_low, ci_high = get_ks_score_confidence_interval_for_binary_classifier(
        y_numerical_true=data[y_target_numerical_col_name],
        y_numerical_score_pred=data[y_predict_numerical_col_name],
        log=log,
    )

    if (ci_low is not None) and (ci_high is not None):
        log.info(
            "KS Score Confidence Interval: [{a:.3f} , {b:.3f}]".format(
                a=ci_low, b=ci_high
            )
        )
        metrics["KS Score Confidence Interval"] = [ci_low, ci_high]

    return metrics


def get_evaluation_binary_classification_metrics(
    data,
    y_target_numerical_col_name,
    y_predict_numerical_col_name,
    y_target_class_col_name,
    y_predict_class_col_name,
    x_y_labels,
    y_no_skill_predict_numerical_col_name: str = None,
    y_no_skill_predict_class_col_name: str = None,
    log=None,
):

    get_common_evaluation_binary_classification_metrics(
        data=data,
        y_target_numerical_col_name=y_target_numerical_col_name,
        y_predict_numerical_col_name=y_predict_numerical_col_name,
        y_no_skill_predict_numerical_col_name=y_no_skill_predict_numerical_col_name,
        log=log,
    )

    log.info(
        "Accuracy: {a:.3f}".format(
            a=get_accuracy_score(
                y_numerical_true=data[y_target_class_col_name],
                y_numerical_pred=data[y_predict_class_col_name],
            )
        )
    )

    log.info(
        "Balanced Accuracy: {a:.3f}".format(
            a=get_balanced_accuracy_score(
                y_numerical_true=data[y_target_class_col_name],
                y_numerical_pred=data[y_predict_class_col_name],
            )
        )
    )

    log.info(
        "Classification Report\n"
        + get_classification_report(
            y_categorical_label_true=data[y_target_class_col_name],
            y_categorical_label_pred=data[y_predict_class_col_name],
            target_classes=x_y_labels,
        )
    )


def get_evaluation_scoring_classification_metrics(
    data,
    y_target_numerical_col_name: str,
    y_predict_numerical_col_name: str,
    y_no_skill_predict_numerical_col_name: str = None,
    results_folder: str = None,
    data_identification: str = None,
    log=None,
):

    metrics = get_common_evaluation_binary_classification_metrics(
        data=data,
        y_target_numerical_col_name=y_target_numerical_col_name,
        y_predict_numerical_col_name=y_predict_numerical_col_name,
        y_no_skill_predict_numerical_col_name=y_no_skill_predict_numerical_col_name,
        log=log,
    )

    if results_folder:
        save_dictionary_to_json(
            data=metrics,
            file_path=results_folder + "metrics_" + data_identification + ".json",
            log=log,
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


def plot_common_evaluation_binary_classification_results(
    data,
    y_target_numerical_col_name,
    y_predict_numerical_col_name,
    date_reference_col_name,
    results_folder,
    dataset_identification,
    y_no_skill_predict_numerical_col_name: str = None,
    view_charts: bool = False,
    save_charts: bool = True,
    log=None,
):
    # scatter predicao x target
    log.info("Plotting Scatter Plot of target x predicted")
    file_path = (
        results_folder + "scatter_pred_x_target_" + dataset_identification + ".png"
    )

    plot_scatter_2d(
        data=data,
        x_col_name=y_target_numerical_col_name,
        y_col_name=y_predict_numerical_col_name,
        title="Scatter plot of target x predicted "
        + dataset_identification
        + " dataset",
        marginal_plot=True,
        view_chart=view_charts,
        save_chart=save_charts,
        file_path_image=file_path,
    )

    # Precision Recall Curve
    log.info("Plotting Precision Recall Curve")
    file_path = results_folder + "pr_curve_" + dataset_identification + ".png"

    precision, recall, thresholds = precision_recall_curve(
        y_true=data[y_target_numerical_col_name],
        y_score=data[y_predict_numerical_col_name],
    )

    if y_no_skill_predict_numerical_col_name:

        # calculate the no skill line as the proportion of the positive class
        no_skill = len(data[data[y_target_numerical_col_name] == 1]) / (data.shape[0])
        recall_ns = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        precision_ns = [no_skill for _ in range(len(recall_ns))]

    else:
        precision_ns = None
        recall_ns = None

    plot_precision_recall_curve(
        precision=precision,
        recall=recall,
        precision_no_skill=precision_ns,
        recall_no_skill=recall_ns,
        title="Precision Recall Curve " + dataset_identification + " dataset",
        view_chart=view_charts,
        save_chart=save_charts,
        file_path_image=file_path,
    )

    # PR ao longo do tempo
    log.info("Plotting PR-AUC score over the time")
    file_path = results_folder + "pr_auc_over_time_" + dataset_identification + ".png"
    plot_pr_auc_score_over_time(
        data=data,
        y_true_column_name=y_target_numerical_col_name,
        y_probas_column_name=y_predict_numerical_col_name,
        date_column_name=date_reference_col_name,
        title="PR-AUC Statistic over time " + dataset_identification + " dataset",
        log=log,
    )

    # ROC Curve
    log.info("Plotting ROC Curve")
    file_path = results_folder + "roc_curve_" + dataset_identification + ".png"
    fpr, tpr, thresholds = roc_curve(
        y_true=data[y_target_numerical_col_name],
        y_score=data[y_predict_numerical_col_name],
    )

    if y_no_skill_predict_numerical_col_name:
        fpr_ns, tpr_ns, thresholds_ns = roc_curve(
            y_true=data[y_target_numerical_col_name],
            y_score=data[y_no_skill_predict_numerical_col_name],
        )
    else:
        fpr_ns = None
        tpr_ns = None

    plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        fpr_no_skill=fpr_ns,
        tpr_no_skill=tpr_ns,
        title="ROC Curve " + dataset_identification + " dataset",
        view_chart=view_charts,
        save_chart=save_charts,
        file_path_image=file_path,
    )

    # ROC ao longo do tempo
    log.info("Plotting ROC-AUC score over the time")
    file_path = results_folder + "roc_auc_over_time_" + dataset_identification + ".png"
    plot_roc_auc_score_over_time(
        data=data,
        y_true_column_name=y_target_numerical_col_name,
        y_probas_column_name=y_predict_numerical_col_name,
        date_column_name=date_reference_col_name,
        title="ROC-AUC Statistic over time " + dataset_identification + " dataset",
        log=log,
    )

    # KS curve
    log.info("Plotting KS Curve")
    file_path = results_folder + "ks_curve_" + dataset_identification + ".png"
    plot_ks_statistic(
        data=data,
        y_true_column_name=y_target_numerical_col_name,
        y_probas_column_name=y_predict_numerical_col_name,
        title="KS Statistic Plot " + dataset_identification + " dataset",
        view_chart=view_charts,
        save_chart=save_charts,
        file_path_image=file_path,
    )

    # KS ao longo do tempo
    log.info("Plotting KS Metric over the time")
    file_path = results_folder + "ks_over_time_" + dataset_identification + ".png"
    plot_ks_score_over_time(
        data=data,
        y_true_column_name=y_target_numerical_col_name,
        y_probas_column_name=y_predict_numerical_col_name,
        date_column_name=date_reference_col_name,
        title="KS Statistic over time " + dataset_identification + " dataset",
        log=log,
    )


def plot_evaluation_binary_classification_results(
    data,
    y_target_numerical_col_name,
    y_target_class_col_name,
    y_predict_numerical_col_name,
    y_predict_class_col_name,
    labels,
    results_folder,
    view_charts,
    save_charts,
    log,
    y_no_skill_predict_numerical_col_name: str = None,
):

    plot_common_evaluation_binary_classification_results(
        data=data,
        y_target_numerical_col_name=y_target_numerical_col_name,
        y_predict_numerical_col_name=y_predict_numerical_col_name,
        y_no_skill_predict_numerical_col_name=y_no_skill_predict_numerical_col_name,
        results_folder=results_folder,
        view_charts=view_charts,
        save_charts=save_charts,
        log=log,
    )

    # confusion matrix
    log.info("Plotting confusion matrix")
    file_path = results_folder + "confusion_matrix.png"

    cm = get_confusion_matrix(
        data=data,
        y_categorical_label_true_col_name=y_target_class_col_name,
        y_categorical_label_pred_col_name=y_predict_class_col_name,
        labels=labels,
    )

    plot_confusion_matrix(
        confusion_matrix=cm,
        x_y_labels=labels,
        title="Confusion Matrix",
        view_chart=view_charts,
        save_chart=save_charts,
        file_path_image=file_path,
    )


def plot_evaluation_scoring_classification_results(
    data,
    y_target_numerical_col_name,
    y_predict_numerical_col_name,
    results_folder,
    dataset_identification: str,
    date_reference_column_name: str = None,
    y_no_skill_predict_numerical_col_name: str = None,
    view_charts: bool = False,
    save_charts: bool = True,
    log=None,
):

    # Histograma de score entre as classes
    log.info("Plotting Score Histogram of two classes")
    file_path = results_folder + "histogram_classes_" + dataset_identification + ".png"
    plot_histogram_binary_classes(
        data=data,
        x_col_name=y_predict_numerical_col_name,
        y_target_col_name=y_target_numerical_col_name,
        title="Histogram of scores between classes "
        + dataset_identification
        + " dataset",
        # cut_offs=[0.2, 0.7],
        view_chart=view_charts,
        save_chart=save_charts,
        file_path_image=file_path,
    )

    # Distribuicoes de score entre as classes
    log.info("Plotting Score Density Distribution of two classes")
    file_path = (
        results_folder + "distribution_classes_" + dataset_identification + ".png"
    )
    plot_distribution_binary_classes(
        data=data,
        x_col_name=y_predict_numerical_col_name,
        y_target_col_name=y_target_numerical_col_name,
        title="Distribution of scores between classes "
        + dataset_identification
        + " dataset",
        # cut_offs=[0.2, 0.7],
        view_chart=view_charts,
        save_chart=save_charts,
        file_path_image=file_path,
    )

    plot_common_evaluation_binary_classification_results(
        data=data,
        y_target_numerical_col_name=y_target_numerical_col_name,
        y_predict_numerical_col_name=y_predict_numerical_col_name,
        date_reference_col_name=date_reference_column_name,
        y_no_skill_predict_numerical_col_name=y_no_skill_predict_numerical_col_name,
        dataset_identification=dataset_identification,
        results_folder=results_folder,
        view_charts=view_charts,
        save_charts=save_charts,
        log=log,
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
    y_predict_numerical_column_name: str,
    y_target_numerical_column_name: str,
    y_predict_class_column_name: str,
    y_target_class_column_name: str,
    labels: list[str],
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
        y_target_numerical_col_name=y_target_numerical_column_name,
        y_predict_numerical_col_name=y_predict_numerical_column_name,
        y_target_class_col_name=y_target_class_column_name,
        y_predict_class_col_name=y_predict_class_column_name,
        x_y_labels=labels,
        log=log,
    )

    log.info(
        "=================================================================================="
    )
    log.info("Evaluating binary classification model data visualization...")

    results_folder_charts = results_folder + "charts/"

    plot_evaluation_binary_classification_results(
        data=data,
        y_target_numerical_col_name=y_target_numerical_column_name,
        y_target_class_col_name=y_target_class_column_name,
        y_predict_numerical_col_name=y_predict_numerical_column_name,
        y_predict_class_col_name=y_predict_class_column_name,
        labels=labels,
        results_folder=results_folder_charts,
        view_charts=view_charts,
        save_charts=save_charts,
        log=log,
    )


def evaluate_scoring_classification(
    data: pd,
    y_predict_numerical_column_name: str,
    y_target_numerical_column_name: str,
    date_reference_column_name: str = None,
    dataset_identification: str = None,
    results_folder: str = None,
    view_charts: bool = True,
    save_charts: bool = True,
    log: object = None,
):

    log.subtitle("Evaluating scoring classification model metrics...")

    results_folder_metrics = results_folder + "metrics/"

    # Criando um estimador "no skill" de baseline (y predict = classe de maior amostragem)
    value = data[y_target_numerical_column_name].mode()[0]
    data["no_skill_predict"] = value

    get_evaluation_scoring_classification_metrics(
        data,
        y_target_numerical_col_name=y_target_numerical_column_name,
        y_predict_numerical_col_name=y_predict_numerical_column_name,
        data_identification=dataset_identification,
        y_no_skill_predict_numerical_col_name="no_skill_predict",
        results_folder=results_folder_metrics,
        log=log,
    )

    log.subtitle("Ploting scoring classification results...")

    results_folder_charts = results_folder + "charts/"

    plot_evaluation_scoring_classification_results(
        data=data,
        y_target_numerical_col_name=y_target_numerical_column_name,
        y_predict_numerical_col_name=y_predict_numerical_column_name,
        date_reference_column_name=date_reference_column_name,
        dataset_identification=dataset_identification,
        y_no_skill_predict_numerical_col_name="no_skill_predict",
        results_folder=results_folder_charts,
        view_charts=view_charts,
        save_charts=save_charts,
        log=log,
    )
