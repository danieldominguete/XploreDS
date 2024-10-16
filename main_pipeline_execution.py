"""
XploreDS :: Orquestrador de disparo de pipelines conforme arquivo de configuracao
"""

# Importing the libraries
import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import argparse


# Importando biblioteca Xplore DS
from xplore_ds.environment.environment import XploreDSLocalhost
from xplore_ds.environment.logging import XploreDSLogging
from xplore_ds.data_handler.file import load_dictionary_from_json
from xplore_ds.data_schemas.pipeline_config import (
    PipelineConfig,
    PipelineType,
    PipelineModelTunningConfig,
)


class PipelineExecution:
    def __init__(self, pipeline_configuration: dict):

        self.config = pipeline_configuration

    def run(self):

        # ==================================================================================
        # Setup de ambiente
        # ==================================================================================
        working_folder = Path(__file__).resolve().parents[0]
        script_name = os.path.basename(__file__)

        # Variaveis de ambiente
        load_dotenv()

        # Setup de ambiente de execucao
        env = XploreDSLocalhost(run_folder=working_folder)

        # Setup de estrutura de logs
        log = XploreDSLogging(project_root=working_folder, script_name=script_name)

        # Cabeçalho de execucao
        log.init_run()
        log.log_environment_setup()

        # ===========================================================================================
        # Orquestrando do tipo de pipeline
        # ==================================================================================

        # Validando leitura do arquivo de configuracao
        if self.config is None:
            raise Exception("Config file is empty")

        # Validando configuracao geral do pipeline
        if "pipeline_config" not in self.config:
            raise Exception("Pipeline configuration not found")
        else:
            pipeline_config = PipelineConfig(**self.config.get("pipeline_config"))

        # Encaminhando job para o pipeline correto
        if pipeline_config.pipeline_type == PipelineType.model_tunning:

            # Validando configuracao do pipeline
            if "pipeline_model_tunning_config" not in self.config:
                raise Exception("Model tunning configuration not found")
            else:
                pipeline_model_tunning_config = PipelineModelTunningConfig(
                    **self.config.get("pipeline_model_tunning_config")
                )

            from xplore_ds.pipelines.model_tunning_pipeline_execution import (
                ModelTunningPipelineExecution,
            )

            # Instanciando pipeline
            pipeline = ModelTunningPipelineExecution(
                config=pipeline_model_tunning_config,
                env=env,
                log=log,
            )

            # Executando pipeline
            pipeline.run()

        # ===========================================================================================
        # Encerrando execucao
        # ==================================================================================
        log.close_run()


# ===========================================================================================
# ===========================================================================================
# Chamada deste script via terminal
if __name__ == "__main__":
    """
    Call from terminal command
    """

    try:
        # recuperando argumentos do script
        parser = argparse.ArgumentParser(description="XploreDS - Pipeline Execution")
        parser.add_argument(
            "-f",
            "--config_file_json",
            help="Json config file for pipeline execution",
            required=True,
        )
        args = parser.parse_args()

        # carregando arquivo de configuracao
        pipe_config_data = load_dictionary_from_json(path_file=args.config_file_json)

        # disparando execucao
        processor = PipelineExecution(pipeline_configuration=pipe_config_data)
        processor.run()
    except Exception as e:
        logging.error(f"{str(e)}")
        raise
# ===========================================================================================
# ===========================================================================================
