"""
Orquestrador de disparo de pipelines conforme arquivo de configuracao
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
        PYTHON_WARNINGS = os.getenv("PYTHON_WARNINGS")

        # Setup de ambiente de execucao
        env = XploreDSLocalhost(run_folder=working_folder)

        # Setup de estrutura de logs
        log = XploreDSLogging(project_root=working_folder, script_name=script_name)

        # Cabeçalho de execucao
        log.init_run()
        log.log_environment_setup()

        # # ===========================================================================================
        # # Script Setup

        # # Loading json file
        # data_config = Util.load_parameters_from_file(path_file=self.parameters_file)

        # # Validate parameters and load environment class
        # env_param = EnvironmentParameters(**data_config.get("environment_parameters"))
        # env = Environment(param=env_param)

        # # Validade parameters and load data processing class
        # data_param = Static2ClassParameters(**data_config.get("static2class_parameters"))
        # ds = DataProcessing(param=data_param)

        # # ===========================================================================================
        # # Setup environment
        # env.init_script(script_name=os.path.basename(__file__), warnings_level=PYTHON_WARNINGS)
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
        logging.error(f"Error: {str(e)}")
        raise
# ===========================================================================================
# ===========================================================================================
