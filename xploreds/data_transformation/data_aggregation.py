import pandas as pd

"""
Xplore DS :: Aggregating Variables
"""

import pandas as pd
from pathlib import Path
import sys
import numpy as np

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))


def generate_features_by_entity_aggregation_for_numerical_variables(
    data: pd.DataFrame,
    id_data_entity_column_name: str,
    numerical_variables_columns_names: list = [],
    log: object = None,
) -> pd.DataFrame:

    # final dataframes
    id_list = data[id_data_entity_column_name].unique().tolist()
    data_agg_final = pd.DataFrame({id_data_entity_column_name: id_list})

    for var in numerical_variables_columns_names:
        if var not in data.columns:
            raise ValueError(f"Column {var} not found in DataFrame")

        try:
            if log:
                log.info(f"Generating simple statistics features of {var} ...")

            # calculando as features de calculos built-in
            data_agg = data.groupby(id_data_entity_column_name).agg(
                {
                    var: [
                        "mean",
                        "median",
                        "min",
                        "max",
                        "std",
                        "sum",
                    ]
                }
            )

            data_agg.columns = [
                "_".join(col).strip() for col in data_agg.columns.values
            ]
            data_agg.reset_index(inplace=True)

            # agregando as features de calculos built-in
            data_agg_final = pd.merge(
                data_agg_final, data_agg, on=id_data_entity_column_name, how="left"
            )
            # calculando as features de calculos customizados
            data_agg = data.groupby(id_data_entity_column_name)[var].apply(
                custom_non_timing_numerical_aggregations
            )

            data_agg = data_agg.unstack()

            data_agg.columns = [var + "_" + col for col in data_agg.columns.values]

            # agregando as features de calculos customizados
            data_agg_final = pd.merge(
                data_agg_final, data_agg, on=id_data_entity_column_name, how="left"
            )

        except Exception as e:
            if log:
                log.error(f"Error generating primitive numerical features: {str(e)}")
            raise

        if log:
            n_features = data_agg.shape[1] - 1
            log.info(f"Total of {n_features} features generated")

    return data_agg_final


def generate_features_by_entity_aggregation_for_categorical_variables(
    data: pd.DataFrame,
    id_data_entity_column_name: str,
    categorical_variables_columns_names: list = [],
    log: object = None,
) -> pd.DataFrame:

    # final dataframes
    id_list = data[id_data_entity_column_name].unique().tolist()
    data_agg_final = pd.DataFrame({id_data_entity_column_name: id_list})

    for var in categorical_variables_columns_names:
        if var not in data.columns:
            raise ValueError(f"Column {var} not found in DataFrame")

        try:
            if log:
                log.info(f"Generating primitive categorical features of {var} ...")

            data_agg = data.groupby(id_data_entity_column_name).agg(
                {
                    var: [
                        "count",
                        "nunique",
                    ]
                }
            )

            data_agg.columns = [
                "_".join(col).strip() for col in data_agg.columns.values
            ]
            data_agg.reset_index(inplace=True)

        except Exception as e:
            if log:
                log.error(f"Error generating primitive categorical features: {str(e)}")
            raise

        data_agg_final = pd.merge(
            data_agg_final, data_agg, on=id_data_entity_column_name, how="left"
        )
    return data_agg_final


def generate_features_by_entity_aggregation_for_datetime_variables(
    data: pd.DataFrame,
    id_data_entity_column_name: str,
    datetime_columns: list = [],
    log: object = None,
) -> pd.DataFrame:

    id_list = data[id_data_entity_column_name].unique().tolist()
    data_agg_final = pd.DataFrame({id_data_entity_column_name: id_list})

    for var in datetime_columns:
        if var not in data.columns:
            raise ValueError(f"Column {var} not found in DataFrame")

        try:
            if log:
                log.info(f"Generating datetime features for {var}")

            data_agg = data.groupby(id_data_entity_column_name).agg(
                {
                    var: [
                        "min",
                        "max",
                        "count",
                        "nunique",
                    ]
                }
            )

            data_agg.columns = [
                "_".join(col).strip() for col in data_agg.columns.values
            ]
            data_agg.reset_index(inplace=True)

            data_agg_final = pd.merge(
                data_agg_final, data_agg, on=id_data_entity_column_name, how="left"
            )

            # customizations
            data_agg = data.groupby(id_data_entity_column_name)[var].apply(
                custom_non_timing_datetime_aggregations
            )

            data_agg = data_agg.unstack()

            data_agg.columns = [var + "_" + col for col in data_agg.columns.values]

            data_agg_final = pd.merge(
                data_agg_final, data_agg, on=id_data_entity_column_name, how="left"
            )

        except Exception as e:
            if log:
                log.error(f"Error generating datetime features: {str(e)}")
            raise

    return data_agg_final


def custom_non_timing_numerical_aggregations(x):
    # Do calculations in one pass through the data
    values = x.values
    return pd.Series(
        {"skew": feature_skew(values), "kurtosis": feature_kurtosis(values)}
    )


def custom_non_timing_datetime_aggregations(x):
    values = x
    return pd.Series(
        {
            "weekday_count": feature_weekday_count(values),
            "weekend_count": feature_weekend_count(values),
        }
    )


def generate_features_by_entity_aggregation_and_timing_references_for_numerical_variables(
    data: pd.DataFrame,
    id_entity_reference_column_name: str,
    feature_datetime_reference_column_name: str,
    datetime_pre_summarization_step_unit: str,
    raw_data_datetime_reference_column_name: str,
    raw_data_past_steps_window_from_reference: int,
    numerical_variables_columns_names: list = [],
    log: object = None,
) -> pd.DataFrame:

    # final dataframes
    data_agg_final = (
        data.groupby(
            [id_entity_reference_column_name, feature_datetime_reference_column_name]
        )
        .size()
        .reset_index()
    )

    for var in numerical_variables_columns_names:
        if var not in data.columns:
            raise ValueError(f"Column {var} not found in DataFrame")

        try:
            if log:
                log.info(f"Generating timing features of {var} ...")

            # pre sumarizacao de dados na unidade basica de tempo
            # truncando o timestamp na granularidade desejada para futura agregacao nao ordenada
            if datetime_pre_summarization_step_unit == "M":
                data[raw_data_datetime_reference_column_name + "_trunc"] = (
                    data[raw_data_datetime_reference_column_name]
                    .dt.to_period("M")
                    .dt.to_timestamp()
                )

            # agregando as datas truncadas em valores de sumarizacao
            data_pre_sum = data.groupby(
                id_entity_reference_column_name,
                feature_datetime_reference_column_name,
                raw_data_datetime_reference_column_name + "_trunc",
            ).agg(
                {
                    var: [
                        "sum",
                    ]
                }
            )

            data_pre_sum.columns = [
                "_".join(col).strip() for col in data_pre_sum.columns.values
            ]
            data_pre_sum.reset_index(inplace=True)

        except Exception as e:
            if log:
                log.error(f"Error generating features: {str(e)}")
            raise

    return data_agg_final


def feature_skew(values):
    # sample skewness of a data set
    from scipy.stats import skew

    return skew(values)


def feature_kurtosis(values):
    # Compute the kurtosis (Fisher or Pearson) of a dataset
    from scipy.stats import kurtosis

    return kurtosis(values)


def feature_weekday_count(values):
    value = (values.dt.weekday < 5).sum()
    return value


def feature_weekend_count(values):
    value = (values.dt.weekday >= 5).sum()
    return value


def feature_morning_count(values):
    value = (values.dt.hour < 12).sum()
    return value
