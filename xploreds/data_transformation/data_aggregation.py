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


def generate_primitive_numerical_features(
    data: pd.DataFrame,
    id_data_entity_column_name: str,
    numerical_features_columns_names: list = [],
    log: object = None,
) -> pd.DataFrame:

    # final dataframes
    id_list = data[id_data_entity_column_name].unique().tolist()
    data_agg_final = pd.DataFrame({id_data_entity_column_name: id_list})

    for var in numerical_features_columns_names:
        if var not in data.columns:
            raise ValueError(f"Column {var} not found in DataFrame")

        try:
            if log:
                log.info(f"Generating primitive numerical features of {var} ...")

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

        except Exception as e:
            if log:
                log.error(f"Error generating primitive numerical features: {str(e)}")
            raise

        data_agg_final = pd.merge(
            data_agg_final, data_agg, on=id_data_entity_column_name, how="left"
        )
    return data_agg_final


def generate_primitive_categorical_features(
    data: pd.DataFrame,
    id_data_entity_column_name: str,
    categorical_features_columns_names: list = [],
    log: object = None,
) -> pd.DataFrame:

    # final dataframes
    id_list = data[id_data_entity_column_name].unique().tolist()
    data_agg_final = pd.DataFrame({id_data_entity_column_name: id_list})

    for var in categorical_features_columns_names:
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


def generate_primitive_datetime_features(
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

        except Exception as e:
            if log:
                log.error(f"Error generating datetime features: {str(e)}")
            raise

    return data_agg_final


def custom_numerical_aggregations(x):
    # Do calculations in one pass through the data
    values = x.values
    return pd.Series({"calc1": custom_calc1(values), "calc2": custom_calc2(values)})


def custom_calc1(values):
    # Do calculations on the values
    return np.mean(values)


def custom_calc2(values):
    # Do calculations on the values
    return np.median(values)


def generate_custom_numerical_features(
    data: pd.DataFrame,
    id_data_entity_column_name: str,
    numerical_features_columns_names: list = [],
    log: object = None,
) -> pd.DataFrame:

    # final dataframes
    id_list = data[id_data_entity_column_name].unique().tolist()
    data_agg_final = pd.DataFrame({id_data_entity_column_name: id_list})

    for var in numerical_features_columns_names:
        if var not in data.columns:
            raise ValueError(f"Column {var} not found in DataFrame")

        try:
            if log:
                log.info(f"Generating custom numerical features of {var} ...")

            data_agg = data.groupby(id_data_entity_column_name)[var].apply(
                custom_numerical_aggregations
            )

            data_agg = data_agg.unstack()

            data_agg.columns = [var + "_" + col for col in data_agg.columns.values]

        except Exception as e:
            if log:
                log.error(f"Error generating custom numerical features: {str(e)}")
            raise

        data_agg_final = pd.merge(
            data_agg_final, data_agg, on=id_data_entity_column_name, how="left"
        )
    return data_agg_final


def custom_datetime_aggregations(x):
    values = x
    return pd.Series(
        {
            "weekday_count": feat_weekday_count(values),
            "weekend_count": feat_weekend_count(values),
        }
    )


def feat_weekday_count(values):
    value = (values.dt.weekday < 5).sum()
    return value


def feat_weekend_count(values):
    value = (values.dt.weekday >= 5).sum()
    return value


def feat_morning_count(values):
    value = (values.dt.hour < 12).sum()
    return value


def generate_custom_datetime_features(
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
                log.info(f"Generating custom datetime features for {var}")

            data_agg = data.groupby(id_data_entity_column_name)[var].apply(
                custom_datetime_aggregations
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
