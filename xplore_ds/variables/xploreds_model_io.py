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
from xplore_ds.data_schemas.model_io_config import (
    ScalingMethod,
    VariableConfig,
)
from xplore_ds.variables.variables_scaling import (
    scaler_variable_fit,
    scaler_variable_transform,
)


class XploreDSModelVariable:
    """
    Classe de variaveis para modelos de machine learning.
    """

    def __init__(
        self,
        name: str = None,
        scaling_method: ScalingMethod = ScalingMethod.none_scaler,
        log: object = None,
    ) -> None:

        self.name = name
        self.name_scaled = None
        self.scaling_method = scaling_method
        self.scaler = None
        self.log = log

    def fit_transform_scaling(self, data):
        """
        Fit & transform variable
        """

        self.scaler = scaler_variable_fit(
            data,
            variable_column_name=self.name,
            scale_method=self.scaling_method,
            log=self.log,
        )

        data, self.name_scaled = scaler_variable_transform(
            data=data,
            feature_column_name=self.name,
            scaler=self.scaler,
            log=self.log,
        )

        return data

    def transform_scaling(self, data):
        """
        Fit variable
        """

        data, self.name_scaled = scaler_variable_transform(
            data=data,
            feature_column_name=self.name,
            scaler=self.scaler,
            log=self.log,
        )

        return data


class XploreDSModelIO:
    """
    Classe de features e target para modelos de machine learning.
    """

    def __init__(
        self,
        features_config: VariableConfig = None,
        target_config: VariableConfig = None,
        log: object = None,
    ) -> None:

        self.log = log
        self.features = {}
        self.target = {}
        self.features_encoded_scaled = []
        self.target_encoded_scaled = []

        for f in features_config:
            self.features[f.name] = XploreDSModelVariable(
                name=f.name,
                scaling_method=f.scaling_method,
                log=self.log,
            )

        for f in target_config:
            self.target[f.name] = XploreDSModelVariable(
                name=f.name,
                scaling_method=f.scaling_method,
                log=self.log,
            )

    def fit_transform(self, data, log=None):
        """
        Fit and transform features and target with scaling
        """

        if log:
            log.info("Fit and transform features...")

        for f in self.features.values():
            if log:
                log.info(
                    "Fitting and transform feature "
                    + f.name
                    + " with "
                    + f.scaling_method
                    + "..."
                )
            data = f.fit_transform_scaling(data)

        if log:
            log.info("Fitting target...")
        for f in self.target.values():
            if log:
                log.info(
                    "Fitting target " + f.name + " with " + f.scaling_method + "..."
                )
            data = f.fit_transform_scaling(data)

        return data

    def transform(self, data, log=None):
        """
        Transform features and target
        """
        if log:
            log.info("Transforming features...")
        for f in self.features.values():
            if log:
                log.info(
                    "Transforming feature "
                    + f.name
                    + " with "
                    + f.scaling_method
                    + "..."
                )
            data = f.transform_scaling(data)

        if log:
            log.info("Transforming target...")
        for f in self.target.values():
            if log:
                log.info(
                    "Transforming target "
                    + f.name
                    + " with "
                    + f.scaling_method
                    + "..."
                )
            data = f.transform_scaling(data)

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
        return list(self.target.keys())

    def get_target_names_scaled(self):
        """
        Get target names scaled
        """
        return [f.name_scaled for f in self.target.values()]
