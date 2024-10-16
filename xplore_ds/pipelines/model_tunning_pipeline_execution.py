"""
Xplore DS : Pipeline de tunning
"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import argparse
from pathlib import Path

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
# Futuramente substituir pois a lib estará já instalada no .venv
project_folder = Path(__file__).resolve().parents[1]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xplore_ds.environment.environment import XploreDSLocalhost
from xplore_ds.environment.logging import XploreDSLogging
from xplore_ds.data_handler.file import (
    load_dataframe_from_parquet,
    save_dataframe_to_parquet,
)
from xplore_ds.models.logistic_regression import XLogisticRegression
from xplore_ds.data_schemas.logistic_regression_config import (
    LogisticRegressionConfig,
    LogisticRegressionHyperparameters,
    Topology,
    FitAlgorithm,
)
from xplore_ds.data_schemas.model_io_config import (
    ModelIOConfig,
    VariableIOConfig,
    ScalingMethod,
    ApplicationType,
)


class ModelTunningPipelineExecution:

    def __init__(
        self,
        config: dict,
        env: XploreDSLocalhost,
        log: XploreDSLogging,
    ):

        self.config = config
        self.env = env
        self.log = log

    def run(self):

        # ----------------------------------------------------------------------------------
        # Setup do modelo

        model_config = LogisticRegressionConfig(
            set_intersection_with_zero=False, topology=Topology.logit
        )

        # ----------------------------------------------------------------------------------
        # Hiperparametros

        tunning_config = LogisticRegressionHyperparameters(
            fit_algorithm=FitAlgorithm.maximum_likelihood,
        )

        # ----------------------------------------------------------------------------------
        # Configuracao de artefatos de saida

        results_folder = self.log.log_path

        output_dataset_train_predict_file_path = (
            results_folder + "data/wine_quality_train_classification_predict.parquet"
        )
        output_dataset_test_predict_file_path = (
            results_folder + "data/wine_quality_test_classification_predict.parquet"
        )
        output_model_file_path = (
            results_folder + "models/wine_quality_logistic_regression.joblib"
        )

        view_charts = True
        save_charts = True

        # **********************************************************************************
        # Execucao do script
        # **********************************************************************************

        # ==================================================================================
        # Carregando base de dados
        # ==================================================================================

        self.log.title("Loading datasets")

        data_train = load_dataframe_from_parquet(
            file_path=self.config.input_dataset_train_file_path, log=self.log
        )

        # ==================================================================================
        # Regras de negócio
        # ==================================================================================

        self.log.title("Training model")

        # ----------------------------------------------------------------------------------
        # Criando topologia do modelo

        self.log.info("Creating model topology...")

        model = XLogisticRegression(
            model_io_config=self.config.model_io_config,
            model_config=model_config,
            tunning_config=tunning_config,
            # random_state=random_state,
            log=self.log,
        )

        # ----------------------------------------------------------------------------------
        # Realizando do tunning do modelo

        self.log.title("Training model")

        model.fit(data=data_train)

        # ----------------------------------------------------------------------------------
        # Apresentando resumo do tunning do modelo
        self.log.title("Summary of tunning")

        model.summary()

        # ----------------------------------------------------------------------------------
        # Avaliando performance do modelo na base de treinamento

        self.log.title("Evaluating model with training data")

        data_train = load_dataframe_from_parquet(
            file_path=self.config.input_dataset_train_file_path, log=self.log
        )

        data_train = model.predict(
            data=data_train,
            y_predict_column_name_output="output_predict_value",
        )

        data_train = model.predict_class(
            data=data_train,
            trigger=0.5,
            y_predict_class_column_name_output="output_predict_class",
        )

        model.evaluate(
            data=data_train,
            y_predict_column_name="output_predict_value",
            y_target_column_name=self.config.model_io_config.target_numerical[0].name,
            view_charts=view_charts,
            save_charts=save_charts,
            results_folder=results_folder,
        )

        # ----------------------------------------------------------------------------------
        # Avaliando performance do modelo na base de teste

        self.log.title("Evaluating model with test data")

        data_test = load_dataframe_from_parquet(
            file_path=self.config.input_dataset_test_file_path, log=self.log
        )
        data_test["target_label"] = np.where(data_test["quality"] <= 5, "bad", "good")

        data_test = model.predict(
            data=data_test,
            y_predict_column_name_output="output_predict",
        )

        model.evaluate(
            data=data_test,
            y_predict_column_name="output_predict",
            y_target_column_name=self.config.model_io_config.target_numerical[0].name,
            view_charts=view_charts,
            save_charts=save_charts,
            results_folder=results_folder,
        )

        # ==================================================================================
        # Salvando artefatos de saida
        # ==================================================================================

        self.log.title("Saving output artifacts")

        model.save(path=output_model_file_path)

        save_dataframe_to_parquet(
            data=data_train,
            file_path=output_dataset_train_predict_file_path,
            log=self.log,
        )

        save_dataframe_to_parquet(
            data=data_test,
            file_path=output_dataset_test_predict_file_path,
            log=self.log,
        )
