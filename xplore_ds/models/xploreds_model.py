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
from xplore_ds.variables.xploreds_model_io import XploreDSModelIO
from xplore_ds.models.evaluate_model import (
    evaluate_regression,
    evaluate_binary_classification,
)
from xplore_ds.data_schemas.model_io_config import ApplicationType, ModelIOConfig


class XploreDSModel(ABC):
    """
    Classe abstrata para modelos de machine learning.
    """

    def __init__(
        self,
        model_io_config: ModelIOConfig = None,
        model_config: BaseModel = None,
        tunning_config: BaseModel = None,
        random_state: int = None,
        log: object = None,
    ) -> None:

        self.model_io_config = model_io_config
        self.model_config = model_config
        self.tunning_config = tunning_config
        self.log = log
        self.model = None
        self.random_state = random_state

        # configurando objeto de io variables
        self.model_io_setup = XploreDSModelIO(
            features_config=self.model_io_config.features,
            target_config=self.model_io_config.target,
            log=self.log,
        )

        super().__init__()

    @abstractmethod
    def fit(self, data):
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

    def evaluate(
        self,
        data,
        y_predict_column_name,
        y_target_column_name,
        y_predict_class_column_name=None,
        y_target_class_column_name=None,
        view_charts=True,
        save_charts=True,
        results_folder=None,
    ):
        """
        Evaluate the model's performance based on application type.
        """

        if self.model_io_config.application_type == ApplicationType.regression:
            evaluate_regression(
                data=data,
                y_predict_column_name=y_predict_column_name,
                y_target_column_name=y_target_column_name,
                view_charts=view_charts,
                save_charts=save_charts,
                results_folder=results_folder,
                log=self.log,
            )
        elif (
            self.model_io_config.application_type
            == ApplicationType.binary_classification
        ):
            evaluate_binary_classification(
                data=data,
                y_predict_column_name=y_predict_column_name,
                y_target_column_name=y_target_column_name,
                view_charts=view_charts,
                save_charts=save_charts,
                results_folder=results_folder,
                log=self.log,
            )
        else:
            self.log.error("Application type not supported")

    def save(self, path):
        """
        Save the trained model to a file.
        """

        # verificando se a pasta existe caso contrario criar a pasta
        create_folder(os.path.dirname(path))

        joblib.dump(self, path)

        self.log.info("Model saved sucessfully in " + path)

    @staticmethod
    def load(path, log):
        """
        Load a trained model from a file.
        """
        model = joblib.load(path)

        log.info("Model loaded sucessfully from " + path)

        return model

    @abstractmethod
    def summary(self):
        """
        Print a tunning results summary.
        """
        pass

    def features_transform(self, data):
        """
        Scale the features of the given data using the trained scaler.
        """

        self.model_io_setup.transform(
            data=data,
            log=self.log,
        )

        return data

    def model_io_fit_transform(self, data):
        """
        Encode and scale the features and target of the given data.
        """

        data = self.model_io_setup.fit_transform(
            data=data,
            log=self.log,
        )

        return data
