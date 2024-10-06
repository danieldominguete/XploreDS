"""
Xplore DS :: Scaling Features
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path
import sys, os

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xplore_ds.data_schemas.knowledge_base_config import ScalingMethod


def scaler_feature_fit(
    data: pd, feature_column_name: str, scale_method: ScalingMethod, log: object
):

    if scale_method == ScalingMethod.mean_std_scaler:
        scaler = StandardScaler()
    elif scale_method == ScalingMethod.min_max_scaler:
        scaler = MinMaxScaler()
    else:
        scaler = None

    scaler.fit(data[feature_column_name].values.reshape(-1, 1))

    return scaler


def scaler_feature_transform(data: pd, feature_column_name: str, scaler, log: object):

    data = scaler.transform(data[feature_column_name].values.reshape(-1, 1))

    return data
