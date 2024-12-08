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
from sklearn.metrics import precision_recall_curve, roc_curve
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr, spearmanr
from scipy.stats import f_oneway, pointbiserialr

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xploreds.data_handler.file import create_folder


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

    # verificando padrao de nomenclatura de classes
    if len(np.unique(y_numerical_true)) != 2:
        raise ValueError(
            "Cannot calculate KS statistic for data with "
            "{} category/ies".format(len(np.unique(y_numerical_true)))
        )

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


def get_ks_score_from_numerical_covariables(
    data: pd,
    y_true_column_name: str,
    covariables_column_name_list: str,
    log: object = None,
):

    ks_values = []

    for v in covariables_column_name_list:

        ks_value = get_ks_score_for_binary_classifier(
            y_numerical_true=data[y_true_column_name],
            y_numerical_score_pred=data[v],
        )

        ks_values.append(ks_value)

        if log:
            log.info("KS: {} = {:.4f}".format(v, ks_value))

    response = pd.DataFrame(
        {
            "variable": covariables_column_name_list,
            "ks": ks_values,
        }
    )

    response.sort_values(by="ks", ascending=True, inplace=True)

    return response


def _iv_discriminatory_analysis(x):

    if x < 0.02:
        return "useless"
    elif x < 0.1:
        return "weak"
    elif x < 0.3:
        return "medium"
    elif x < 0.5:
        return "strong"
    else:
        return "suspicious"


def get_information_value(
    data: pd,
    y_true_numeric_column_name,
    var_categorical_column_name,
    var_numeric_column_names,
    n_numerical_bins=10,
    log=None,
):

    if len(data[y_true_numeric_column_name].unique()) != 2:
        raise ValueError(
            "Cannot calculate IV for data with "
            "{} category/ies".format(len(data[y_true_numeric_column_name].unique()))
        )
    var_list = var_categorical_column_name + var_numeric_column_names
    df_iv = pd.DataFrame(columns=["variable", "iv"])
    df_woe = pd.DataFrame(columns=["variable", "variable_group", "woe", "iv"])

    for var in var_list:

        if var in var_categorical_column_name:
            data_temp = data[[var, y_true_numeric_column_name]]

        if var in var_numeric_column_names:
            data_temp = data[[var, y_true_numeric_column_name]]
            data_temp[var] = pd.qcut(
                data_temp[var], q=n_numerical_bins, duplicates="drop", precision=0
            )

        data_temp = data_temp.astype({var: str})
        data_agg = data_temp.groupby(var, as_index=False, dropna=False).agg(
            {y_true_numeric_column_name: ["count", "sum"]}
        )
        data_agg.columns = ["variable_group", "count", "events"]
        data_agg["variable"] = var
        data_agg["non_events"] = data_agg["count"] - data_agg["events"]
        data_agg["perc_events"] = data_agg["events"] / data_agg["count"]
        data_agg["perc_non_events"] = data_agg["non_events"] / data_agg["count"]
        data_agg["woe"] = np.log(data_agg["perc_events"] / data_agg["perc_non_events"])
        data_agg = data_agg.replace({"woe": {np.inf: 0, -np.inf: 0}})
        data_agg["iv"] = (
            data_agg["perc_events"] - data_agg["perc_non_events"]
        ) * data_agg["woe"]

        df_woe = pd.concat(
            [df_woe, data_agg[["variable", "variable_group", "woe", "iv"]]]
        )
        df_iv = pd.concat(
            [
                df_iv,
                pd.DataFrame(
                    {
                        "variable": [var],
                        "iv": [data_agg["iv"].sum()],
                    }
                ),
            ]
        )

        if log:
            log.info("IV of variable {} = {:.2f}".format(var, data_agg["iv"].sum()))

    df_iv["analysis"] = df_iv["iv"].apply(lambda x: _iv_discriminatory_analysis(x))
    df_iv.sort_values(by="iv", ascending=True, inplace=True)

    # identificador de variavel + categoria
    df_woe["variable_group_full"] = (
        df_woe["variable"].astype(str) + "_" + df_woe["variable_group"].astype(str)
    )

    df_woe.sort_values(by="woe", ascending=False, inplace=True)

    return df_iv, df_woe


def get_association_statistics(
    data,
    numerical_variables: list = None,
    categorical_variables: list = None,
    log: object = None,
):

    var_1 = []
    var_2 = []
    metric = []
    value = []
    p_value = []

    # numerical x numerical
    if log:
        log.info("Numerical variables association metrics calculation...")
    for n1 in numerical_variables:
        for n2 in numerical_variables:

            # pearson
            var_1.append(n1)
            var_2.append(n2)
            metric.append("pearson")
            # pearson_metric = data[[n1, n2]].corr(method="pearson").iloc[0, 1]
            pearson_metric, pearson_p_value = pearsonr(data[n1], data[n2])
            value.append(pearson_metric)
            p_value.append(pearson_p_value)

            # spearman
            var_1.append(n1)
            var_2.append(n2)
            metric.append("spearman")
            # spearman_metric = data[[n1, n2]].corr(method="spearman").iloc[0, 1]
            spearman_metric, spearman_p_value = spearmanr(data[n1], data[n2])
            value.append(spearman_metric)
            p_value.append(spearman_p_value)

            # kendall tau
            var_1.append(n1)
            var_2.append(n2)
            metric.append("kendall")
            kendall_metric = data[[n1, n2]].corr(method="kendall").iloc[0, 1]
            value.append(kendall_metric)
            p_value.append(None)

    # numerical x categorical
    if log:
        log.info(
            "Numerical and categorical variables association metrics calculation ..."
        )
    for n1 in numerical_variables:
        for n2 in categorical_variables:

            # point biserial (2 classes)
            if len(data[n2].unique()) == 2:
                var_1.append(n1)
                var_2.append(n2)
                metric.append("point biserial")
                point_biserial_metric, point_biserial_p_value = pointbiserialr(
                    data[n2], data[n1]
                )
                value.append(point_biserial_metric)
                p_value.append(point_biserial_p_value)

                # repetindo o registro para similiaridade na matriz
                var_2.append(n1)
                var_1.append(n2)
                metric.append("point biserial")
                value.append(point_biserial_metric)
                p_value.append(point_biserial_p_value)

            # ANOVA (3 ou mais classes)
            elif len(data[n2].unique()) > 2:

                var_1.append(n1)
                var_2.append(n2)
                metric.append("anova")
                groups = [
                    data[data[n2] == category][n1] for category in data[n2].unique()
                ]
                anova_metric, anova_p_value = f_oneway(*groups)
                value.append(anova_metric)
                p_value.append(anova_p_value)

                # repetindo o registro para similiaridade na matriz
                var_2.append(n1)
                var_1.append(n2)
                metric.append("anova")
                value.append(anova_metric)
                p_value.append(anova_p_value)

    # categorical x categorical
    if log:
        log.info("Categorical variables association metrics calculation ...")
    for n1 in categorical_variables:
        for n2 in categorical_variables:

            # Cramers's V
            var_1.append(n1)
            var_2.append(n2)
            metric.append("cramers v")

            contingency_table = pd.crosstab(data[n1], data[n2])
            chi2, p, dof, ex = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            cramers_metric = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            value.append(cramers_metric)
            p_value.append(None)

    # saving statistics
    response = pd.DataFrame(
        {
            "variable_1": var_1,
            "variable_2": var_2,
            "metric": metric,
            "value": value,
        }
    )
    return response
