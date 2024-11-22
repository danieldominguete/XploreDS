"""
Xplore DS :: Statistics methods
"""

import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, bootstrap
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy.stats import ks_2samp

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))


def get_mean_absolute_error(y_numerical_true, y_numerical_pred):
    return mean_absolute_error(y_true=y_numerical_true, y_pred=y_numerical_pred)


def get_explained_variance_score(y_numerical_true, y_numerical_pred):
    return explained_variance_score(y_true=y_numerical_true, y_pred=y_numerical_pred)


def get_max_error_score(y_numerical_true, y_numerical_pred):
    return max_error(y_true=y_numerical_true, y_pred=y_numerical_pred)


def get_mse_error_score(y_numerical_true, y_numerical_pred):
    return mean_squared_error(y_true=y_numerical_true, y_pred=y_numerical_pred)


def get_mdae_error_score(y_numerical_true, y_numerical_pred):
    return median_absolute_error(y_true=y_numerical_true, y_pred=y_numerical_pred)


def get_r2_score(y_numerical_true, y_numerical_pred):
    return r2_score(y_true=y_numerical_true, y_pred=y_numerical_pred)


def get_accuracy_score(y_numerical_true, y_numerical_pred):
    return accuracy_score(y_true=y_numerical_true, y_pred=y_numerical_pred)


def get_balanced_accuracy_score(y_numerical_true, y_numerical_pred):
    return balanced_accuracy_score(y_true=y_numerical_true, y_pred=y_numerical_pred)


def get_classification_report(y_categorical_label_true, y_categorical_label_pred):
    return classification_report(
        y_true=y_categorical_label_true,
        y_pred=y_categorical_label_pred,
    )


def get_roc_auc_score_for_binary_classifier(y_numerical_true, y_numerical_score_pred):
    return roc_auc_score(y_true=y_numerical_true, y_score=y_numerical_score_pred)


def get_gini_score_for_binary_classifier(y_numerical_true, y_numerical_score_pred):
    auroc = roc_auc_score(y_true=y_numerical_true, y_score=y_numerical_score_pred)
    return 2 * auroc - 1


def get_ks_score_for_binary_classifier(y_numerical_true, y_numerical_score_pred):
    v = ks_2samp(
        y_numerical_score_pred[y_numerical_true == 0],
        y_numerical_score_pred[y_numerical_true == 1],
    )
    return v.statistic


def get_ks_score_confidence_interval_for_binary_classifier(
    y_numerical_true, y_numerical_score_pred
):
    ci_low, ci_high = get_ks_statistics_interval_confidence(
        y_numerical_true, y_numerical_score_pred
    )
    return ci_low, ci_high


def get_precision_recall_score_for_binary_classifier(
    y_numerical_true, y_numerical_score_pred
):
    precision, recall, _ = precision_recall_curve(
        y_numerical_true, y_numerical_score_pred
    )
    return auc(x=recall, y=precision)


def get_confusion_matrix(
    data, y_categorical_label_true_col_name, y_categorical_label_pred_col_name
):

    cm = confusion_matrix(
        y_true=data[y_categorical_label_true_col_name],
        y_pred=data[y_categorical_label_pred_col_name],
        normalize="all",
    )
    return cm


def get_binary_ks_curve(data: pd, y_true_column_name: str, y_probas_column_name: str):

    y_true, y_probas = np.asarray(data[y_true_column_name]), np.asarray(
        data[y_probas_column_name]
    )

    # garantindo marcacao por 0 e 1
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


def get_ks_statistics_interval_confidence(y_numerical_true, y_numerical_score_pred):

    # print(len(y_numerical_score_pred[y_numerical_true == 0]))
    # print(len(y_numerical_score_pred[y_numerical_true == 1]))
    # print(len(np.unique(y_numerical_true)))

    if (len(y_numerical_score_pred[y_numerical_true == 0])) > 2 and (
        (len(y_numerical_score_pred[y_numerical_true == 1]) > 2)
        and (len(np.unique(y_numerical_true)) == 2)
    ):
        res = bootstrap(
            (
                y_numerical_score_pred[y_numerical_true == 0],
                y_numerical_score_pred[y_numerical_true == 1],
            ),
            ks_2samp,
            vectorized=False,
            paired=False,
            n_resamples=1000,
            confidence_level=0.95,
            random_state=42,
        )

        return res.confidence_interval.low[0], res.confidence_interval.high[0]
    else:
        return None, None


def get_ks_score_over_time(
    data: pd,
    y_true_column_name: str,
    y_probas_column_name: str,
    time_column_name: str,
    log: object = None,
):

    time_frame = list(data[time_column_name].unique())
    time_frame.sort()

    ks_values = []
    ks_ci_low_values = []
    ks_ci_high_values = []

    for t in time_frame:

        data_temp = data[data[time_column_name] == t]
        ks_value = get_ks_score_for_binary_classifier(
            y_numerical_true=data_temp[y_true_column_name],
            y_numerical_score_pred=data_temp[y_probas_column_name],
        )
        ks_ci_low, ks_ci_high = get_ks_statistics_interval_confidence(
            y_numerical_true=data_temp[y_true_column_name],
            y_numerical_score_pred=data_temp[y_probas_column_name],
        )

        ks_values.append(ks_value)
        ks_ci_low_values.append(ks_ci_low)
        ks_ci_high_values.append(ks_ci_high)

    return time_frame, ks_values, ks_ci_low_values, ks_ci_high_values
