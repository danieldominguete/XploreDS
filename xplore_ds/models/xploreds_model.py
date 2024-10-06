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


class XploreDSModel(ABC):
    """
    Classe abstrata para modelos de machine learning.
    """

    def __init__(
        self,
        setup: BaseModel = None,
        hyperparameters: BaseModel = None,
        log: object = None,
    ) -> None:

        self.model = None
        self.setup = setup
        self.hyperparameters = hyperparameters
        self.log = log

        super().__init__()

    @abstractmethod
    def train(self, data_train, features_column_name, y_target_column_name):
        """
        Train machine learning model
        """
        pass

    @abstractmethod
    def predict(self, data_test, features_column_name, y_predict_column_name):
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
