import pandas as pd

"""
Xplore DS :: Aggregating Variables
"""

import pandas as pd
from pathlib import Path
import sys

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))


def generate_primitive_numerical_features(
    data: pd.DataFrame,
    id_data_entity_column_name: str,
    feature_column_name: str,
    log: object = None,
) -> pd.DataFrame:

    if feature_column_name not in data.columns:
        raise ValueError(f"Column {feature_column_name} not found in DataFrame")

    try:
        if log:
            log.info(
                f"Generating primitive numerical features of {feature_column_name} ..."
            )

        data_agg = data.groupby(id_data_entity_column_name).agg(
            {
                feature_column_name: [
                    "mean",
                    "median",
                    "min",
                    "max",
                    "std",
                    "sum",
                    "count",
                    "nunique",
                ]
            }
        )

        data_agg.columns = ["_".join(col).strip() for col in data_agg.columns.values]
        data_agg.reset_index(inplace=True)

        return data_agg

    except Exception as e:
        if log:
            log.error(f"Error generating primitive numerical features: {str(e)}")
        raise
