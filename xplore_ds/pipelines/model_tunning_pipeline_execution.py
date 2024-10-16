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
    LogisticRegressionArchiteture,
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
from xplore_ds.data_schemas.pipeline_config import ModelType


class ModelTunningPipelineExecution:

    def __init__(
        self,
        config: object,
        env: XploreDSLocalhost,
        log: XploreDSLogging,
        view_charts: bool = False,
        save_charts: bool = True,
    ):

        self.config = config
        self.env = env
        self.log = log
        self.view_charts = view_charts
        self.save_charts = save_charts

    def run(self):

        # ----------------------------------------------------------------------------------
        # Setup da arquitetura e hyperparametros do modelo

        if self.config.model_type == ModelType.logistic_regression:

            model_config = LogisticRegressionArchiteture(
                **self.config.model_architeture
            )

            tunning_config = LogisticRegressionHyperparameters(
                **self.config.model_hyperparameters
            )
        else:
            raise Exception("Model type not supported")

        # ----------------------------------------------------------------------------------
        # Configuracao de artefatos de saida

        results_folder = self.log.log_path

        output_dataset_train_predict_file_path = (
            results_folder + "data/" + self.log.log_run + "_predict_train.parquet"
        )
        output_dataset_test_predict_file_path = (
            results_folder + "data/" + self.log.log_run + "_predict_test.parquet"
        )
        output_model_file_path = (
            results_folder + "models/" + self.log.log_run + "_model.joblib"
        )

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

        self.log.info("Predicting output value ...")
        data_train = model.predict(
            data=data_train,
            y_predict_column_name_output="output_predict_value",
        )

        self.log.info("Predicting output class ...")
        data_train = model.predict_class(
            data=data_train,
            trigger=0.5,
            y_predict_class_column_name_output="output_predict_class",
            int_to_class_map={0: "bad", 1: "good"},
        )

        model.evaluate(
            data=data_train,
            y_predict_column_name="output_predict_value",
            y_target_column_name=self.config.model_io_config.target_numerical[0].name,
            view_charts=self.view_charts,
            save_charts=self.save_charts,
            results_folder=results_folder,
        )

        # ----------------------------------------------------------------------------------
        # Avaliando performance do modelo na base de teste

        self.log.title("Evaluating model with test data")

        data_test = load_dataframe_from_parquet(
            file_path=self.config.input_dataset_test_file_path, log=self.log
        )

        data_test = model.predict(
            data=data_test,
            y_predict_column_name_output="output_predict",
        )

        data_test = model.predict_class(
            data=data_test,
            trigger=0.5,
            y_predict_class_column_name_output="output_predict_class",
        )

        model.evaluate(
            data=data_test,
            y_predict_column_name="output_predict",
            y_target_column_name=self.config.model_io_config.target_numerical[0].name,
            view_charts=self.view_charts,
            save_charts=self.save_charts,
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
