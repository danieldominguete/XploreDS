"""
Xplore DS :: Features Model Class
"""

from abc import ABC, abstractmethod
import joblib
from pathlib import Path
import sys, os
from pydantic import BaseModel

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xplore_ds.data_handler.file import create_folder
from xplore_ds.data_schemas.knowledge_base_config import ScalingMethod, FeaturesConfig
from xplore_ds.features.features_scaling import (
    scaler_feature_fit,
    scaler_feature_transform,
)


class XploreDSFeature:
    """
    Classe de features para modelos de machine learning.
    """

    def __init__(
        self,
        name: str = None,
        scaling_method: ScalingMethod = ScalingMethod.none_scaler,
        log: object = None,
    ) -> None:

        self.name = name

        self.scaling_method = scaling_method
        self.scaler = None
        self.log = log

    def fit(self, data):
        """
        Fit features
        """
        self.scaler = scaler_feature_fit(
            data,
            feature_column_name=self.name,
            scale_method=self.scaling_method,
            log=self.log,
        )

        return self

    def transform(self, data):
        """
        Fit features
        """
        self.name_scaled = self.name + "_scaled"

        data = scaler_feature_transform(
            data=data,
            feature_column_name=self.name,
            feature_column_name_scaled=self.name_scaled,
            scaler=self.scaler,
            log=self.log,
        )

        return data


class XploreDSFeatures:
    """
    Classe de features para modelos de machine learning.
    """

    def __init__(
        self,
        features_config: FeaturesConfig = None,
        log: object = None,
    ) -> None:

        self.log = log
        self.features = {}

        for f in features_config:
            self.features[f.name] = XploreDSFeature(
                name=f.name, scaling_method=f.scaling_method, log=self.log
            )

    def fit(self, data, log=None):
        """
        Fit features
        """
        for f in self.features.values():
            f.fit(data)

    def transform(self, data, log=None):
        """
        Transform features
        """
        for f in self.features.values():
            data = f.transform(data)
        return data

    def save(self, path: str = None, log=None):
        """
        Save features
        """

        create_folder(path)
        joblib.dump(self.features)

    def load(self, path: str = None, log=None):
        """
        Load features
        """
        if path is None:
            path = self.kb_setup.features_setup.path
        self.features = joblib.load(path + self.setup.features_setup.name + ".joblib")

    def get_features_names(self):
        """
        Get features names
        """
        return list(self.features.keys())

    def get_features_names_scaled(self):
        """
        Get features names scaled
        """
        return [f.name_scaled for f in self.features.values()]

    def get_target_name(self):
        """
        Get target name
        """
        return list(self.features.keys())
