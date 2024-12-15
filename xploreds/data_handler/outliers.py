"""
Xplore DS :: Outliers Values Tools Package
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.covariance import EllipticEnvelope

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))


def remove_unidimensional_outliers_by_zscore(
    data: pd.DataFrame,
    column_names: list = None,
    zscore_threshold: float = 3.0,
    log=None,
):

    for var in column_names:

        # Calculando a media e desvio padrao da variavel
        mean = data[var].mean()
        std = data[var].std()

        # Calculando o zscore da variavel
        zscore = (data[var] - mean) / std

        # Removendo outliers
        n0 = data.shape[0]
        data = data[zscore.abs() < zscore_threshold]
        n1 = data.shape[0]

        deleted = n0 - n1
        deleted_percentage = (deleted / n0) * 100

        if log:
            log.info(
                f"Removing outliers from {var} by zscore: {zscore_threshold} (deleted: {deleted}, percentage: {deleted_percentage:.2f}%)"
            )

    return data


def remove_unidimensional_outliers_by_iqr(
    data: pd.DataFrame,
    column_names: list = None,
    iqr_threshold: float = 1.5,
    log=None,
):

    for var in column_names:

        # Calculando o Q1, Q3 e IQR da variavel
        Q1 = data[var].quantile(0.25)
        Q3 = data[var].quantile(0.75)
        IQR = Q3 - Q1

        # Calculando os limites inferior e superior
        lower_limit = Q1 - iqr_threshold * IQR
        upper_limit = Q3 + iqr_threshold * IQR

        # Removendo outliers
        n0 = data.shape[0]
        data = data[(data[var] >= lower_limit) & (data[var] <= upper_limit)]
        n1 = data.shape[0]

        deleted = n0 - n1
        deleted_percentage = (deleted / n0) * 100

        if log:
            log.info(
                f"Removing outliers from {var} by IQR: {iqr_threshold} (deleted: {deleted}, percentage: {deleted_percentage:.2f}%)"
            )

    return data


def replace_unidimensional_outliers_by_winsorizing(
    data: pd.DataFrame,
    column_names: list = None,
    max_percentile_threshold: float = 0.95,
    min_percentile_threshold: float = 0.05,
    log=None,
):

    for var in column_names:

        # Calculando os limites inferior e superior
        lower_limit = data[var].quantile(min_percentile_threshold)
        upper_limit = data[var].quantile(max_percentile_threshold)

        # Substituindo outliers por limites
        data[var] = data[var].clip(lower_limit, upper_limit)

        if log:
            log.info(
                f"Replacing outliers from {var} by winsorizing: {min_percentile_threshold} and {max_percentile_threshold}"
            )

    return data


def remove_unidimensional_outliers_by_winsorizing(
    data: pd.DataFrame,
    column_names: list = None,
    max_percentile_threshold: float = 0.95,
    min_percentile_threshold: float = 0.05,
    log=None,
):
    for var in column_names:

        # Calculando o percentil da variavel
        max_percentile = data[var].quantile(max_percentile_threshold)
        min_percentile = data[var].quantile(min_percentile_threshold)

        # Removendo outliers
        n0 = data.shape[0]
        data = data[(data[var] >= min_percentile) & (data[var] <= max_percentile)]
        n1 = data.shape[0]

        deleted = n0 - n1
        deleted_percentage = (deleted / n0) * 100

        if log:
            log.info(
                f"Removing outliers from {var} by percentile limits: {min_percentile_threshold} and {max_percentile_threshold} (deleted: {deleted}, percentage: {deleted_percentage:.2f}%)"
            )

    return data


def remove_multidimensional_outliers_by_elliptic_envelope(
    data: pd.DataFrame,
    column_names: list = None,
    contamination: float = 0.01,
    log=None,
):

    outlier_detector = EllipticEnvelope(contamination=contamination)
    outlier_detector.fit(data[column_names])

    # Predicting outliers (1 = inlier, -1 = outlier)
    outliers = outlier_detector.predict(data[column_names])
    n0 = data.shape[0]
    data = data[outliers == 1]
    n1 = data.shape[0]

    deleted = n0 - n1
    deleted_percentage = (deleted / n0) * 100

    if log:
        log.info(
            f"Removing outliers from {column_names} by elliptic envelope: {contamination} (deleted: {deleted}, percentage: {deleted_percentage:.2f}%)"
        )

    return data


def remove_multidimensional_outliers_by_isolation_forest(
    data: pd.DataFrame,
    column_names: list = None,
    contamination: float = 0.01,
    log=None,
):

    from sklearn.ensemble import IsolationForest

    outlier_detector = IsolationForest(contamination=contamination)
    outlier_detector.fit(data[column_names])

    # Predicting outliers (1 = inlier, -1 = outlier)
    outliers = outlier_detector.predict(data[column_names])
    n0 = data.shape[0]
    data = data[outliers == 1]
    n1 = data.shape[0]

    deleted = n0 - n1
    deleted_percentage = (deleted / n0) * 100

    if log:
        log.info(
            f"Removing outliers from {column_names} by isolation forest: {contamination} (deleted: {deleted}, percentage: {deleted_percentage:.2f}%)"
        )

    return data
