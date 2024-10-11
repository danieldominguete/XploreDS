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
from xplore_ds.data_schemas.logistic_regression_config import Topology, FitAlgorithm


class XLogisticRegression(XploreDSModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit(
        self,
        data: pd,
    ) -> None:

        # aplicando processamento de encoder e scaling no dataset
        self.log.title("Input and output variables pre-processing...")

        data = self.model_io_fit_transform(
            data=data,
        )

        data_input = data[self.model_io_setup.get_features_names_scaled()]

        # incluindo coeficiente independente
        if self.model_config.set_intersection_with_zero == False:
            data_input = sm.add_constant(data_input)

        if self.tunning_config.fit_algorithm == FitAlgorithm.maximum_likelihood:
            if self.model_config.topology == Topology.logit:
                self.model = sm.Logit(
                    data[self.model_io_setup.get_target_name()], data_input
                )
            elif self.model_config.topology == Topology.probit:
                self.model = sm.Probit(
                    data[self.model_io_setup.get_target_name()], data_input
                )
            else:
                self.log.error("Topology not implemented")
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

        data_input = data[self.model_io_setup.get_features_names_scaled()]

        # incluindo coeficiente independente
        if self.model_config.set_intersection_with_zero == False:
            data_input = sm.add_constant(data_input)

        data[y_predict_column_name] = self.model.predict(data_input)

        return data

    def predict_class(self, data, trigger, y_predict_class_column_name):
        """
        Calculate predicted class for the given test data.
        """

        self.predict(data, "_predicted_value")

        data[y_predict_class_column_name] = data["_predicted_value"].apply(
            lambda x: 1 if x > trigger else 0
        )

        data = data.drop(columns=["_predicted_value"])

        return data
