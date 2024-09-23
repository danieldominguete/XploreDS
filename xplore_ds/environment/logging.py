"""
Xplore DS :: Logging Tools Package
"""

import logging
from datetime import datetime
from pathlib import Path
from xplore_ds.data_handler.file import create_folder


class XploreDSLogging:

    def __init__(self, project_root: str, script_name: str) -> None:

        # capturando referencias de tempos de inicialização
        self.dt_init = datetime.now()
        self.dt_init_str = self.dt_init.strftime("%y%m%d%H%M%S")

        # configurando o nome do arquivo de log
        self.log_name = Path(str(script_name)).stem + "_" + self.dt_init_str + ".log"
        self.log_run = Path(str(script_name)).stem + "_" + self.dt_init_str

        # configurando locais de registros de logs
        self.log_path = str(project_root) + "/runs/" + str(self.log_run) + "/"

        # configurando o path completo de log
        self.log_file = self.log_path + self.log_name

        # criando a pasta caso não exista
        create_folder(self.log_path)

        # configurando o logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> None:

        # configurando o logger
        log_config = {
            "level": logging.INFO,
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "datefmt": "%d-%m-%Y %H:%M:%S",
            "handlers": [logging.FileHandler(self.log_file), logging.StreamHandler()],
        }

        logging.basicConfig(**log_config)

        return logging.basicConfig(**log_config)

    def init_run(self) -> None:

        # configurando abertura de run
        logging.info(
            "=================================================================================="
        )
        logging.info("Iniciando script: " + self.log_run)
        logging.info(
            "=================================================================================="
        )

    def close_run(self) -> None:

        # configurando encerramento de run
        logging.info(
            "=================================================================================="
        )
        logging.info("Finalizando script: " + self.log_run)
        logging.info(
            "=================================================================================="
        )
