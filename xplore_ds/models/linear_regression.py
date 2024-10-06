"""
Xplore DS :: Linear Regression Models
"""

# importando as bibliotecas padrao
from pathlib import Path
import sys
import statsmodels.api as sm
import pandas as pd


# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xplore_ds.models.xploreds_model import XploreDSModel
from xplore_ds.data_schemas.linear_regression_config import FitAlgorithm


class XLinearRegression(XploreDSModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def train(self, data: pd, features_column_name: str, target_column: str) -> None:

        data_input = data[features_column_name]

        # incluindo coeficiente independente
        if self.setup.set_intersection_with_zero == False:
            data_input = sm.add_constant(data_input)

        if self.hyperparameters.fit_algorithm == FitAlgorithm.ordinary_least_squares:
            self.model = sm.OLS(data[target_column], data_input)
        elif (
            self.hyperparameters.fit_algorithm == FitAlgorithm.generalized_least_squares
        ):
            self.model = sm.GLS(data[target_column], data_input)
        else:
            self.log.error("Fit algorithm not implemented")

        self.model = self.model.fit()

    def summary(self):

        self.log.info(self.model.summary())

    def predict(self, data_test, features_column_name, y_predict_column_name):
        """
        Calculate predictions for the given test data.
        """

        # filtrado somente features
        data_input = data_test[features_column_name]

        # incluindo coeficiente independente
        if self.setup.set_intersection_with_zero == False:
            data_input = sm.add_constant(data_input)

        data_test[y_predict_column_name] = self.model.predict(data_input)

        return data_test

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance on the test data.
        """
        pass
