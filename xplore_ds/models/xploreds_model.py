"""
Xplore DS :: General Model Class
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
from xplore_ds.features.xploreds_features import XploreDSFeatures


class XploreDSModel(ABC):
    """
    Classe abstrata para modelos de machine learning.
    """

    def __init__(
        self,
        kb_config: BaseModel = None,
        model_config: BaseModel = None,
        tunning_config: BaseModel = None,
        log: object = None,
    ) -> None:

        self.kb_config = kb_config
        self.model_config = model_config
        self.tunning_config = tunning_config
        self.log = log
        self.model = None

        self.features_setup = XploreDSFeatures(
            features_config=self.kb_config.features,
            log=self.log,
        )

        self.target_setup = XploreDSFeatures(
            features_config=self.kb_config.target,
            log=self.log,
        )

        super().__init__()

    @abstractmethod
    def fit(self, data, random_state, log):
        """
        Train machine learning model
        """
        pass

    @abstractmethod
    def predict(self, data, y_predict_column_name):
        """
        Calculate predictions for the given test data.
        """
        pass

    @abstractmethod
    def evaluate(self, data_eval, y_predict_column_name, y_target_column_name):
        """
        Evaluate the model's performance on the test data.
        """
        pass

    def save(self, path):
        """
        Save the trained model to a file.
        """

        # verificando se a pasta existe caso contrario criar a pasta
        create_folder(os.path.dirname(path))

        joblib.dump(self, path)

    @staticmethod
    def load(path):
        """
        Load a trained model from a file.
        """
        return joblib.load(path)

    @abstractmethod
    def summary(self):
        """
        Print a tunning results summary.
        """
        pass

    def features_fit(self, data):
        """
        Scale the features of the given data using the trained scaler.
        """

        self.features_setup.fit(
            data=data,
            log=self.log,
        )

        return data

    def features_transform(self, data):
        """
        Scale the features of the given data using the trained scaler.
        """

        self.features_setup.transform(
            data=data,
            log=self.log,
        )

        return data

    def features_fit_transform(self, data):
        """
        Scale the features of the given data using the trained scaler.
        """

        self.features_setup.fit(
            data=data,
            log=self.log,
        )

        self.features_setup.transform(
            data=data,
            log=self.log,
        )

        return data
