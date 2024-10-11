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

    def fit(
        self,
        data: pd,
    ) -> None:

        # aplicando processamento de scaling de features
        self.log.title("Features fit")

        data = self.model_io_fit_transform(
            data=data,
        )

        data_input = data[self.features_setup.get_features_names_scaled()]

        # incluindo coeficiente independente
        if self.model_config.set_intersection_with_zero == False:
            data_input = sm.add_constant(data_input)

        if self.tunning_config.fit_algorithm == FitAlgorithm.ordinary_least_squares:
            target_column = self.target_setup.get_target_name()
            self.model = sm.OLS(data[target_column], data_input)
        elif (
            self.tunning_config.fit_algorithm == FitAlgorithm.generalized_least_squares
        ):
            target_column = self.target_setup.get_target_name()
            self.model = sm.GLS(data[target_column], data_input)
        else:
            self.log.error("Fit algorithm not implemented")

        self.model = self.model.fit()

    def summary(self):

        self.log.info(self.model.summary())

    def predict(self, data, y_predict_column_name):
        """
        Calculate predictions for the given test data.
        """

        data = self.features_transform(
            data=data,
        )

        data_input = data[self.features_setup.get_features_names_scaled()]

        # incluindo coeficiente independente
        if self.model_config.set_intersection_with_zero == False:
            data_input = sm.add_constant(data_input)

        data[y_predict_column_name] = self.model.predict(data_input)

        return data
