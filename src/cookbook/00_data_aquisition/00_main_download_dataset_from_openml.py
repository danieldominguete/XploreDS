"""
Xplore DS :: Dataset download from Open ML website
"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import ssl
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[3]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xploreds.environment.environment import XploreDSLocalhost
from xploreds.environment.logging import XploreDSLogging
from xploreds.data_handler.file import save_dataframe_to_parquet
from xploreds.data_handler.dataframe import describe_dataframe, create_unique_id

# ==================================================================================
# Setup do script

script_name = os.path.basename(__file__)

# Variaveis de ambiente
load_dotenv()

# Criando estrutura de execucao local
env = XploreDSLocalhost(run_folder=project_folder)

# Criando estrutura de logs
log = XploreDSLogging(project_root=project_folder, script_name=script_name)
log.init_run()

# Configuracao de SSL
ssl._create_default_https_context = ssl._create_unverified_context

# ==================================================================================
# Parametrizacao do script
# ==================================================================================

log.title("Script setup")

# EXEMPLO DE DATASET DE CLASSIFICACAO BINARIA :: CREDIT RISK
# https://openml.org/search?type=data&status=active&sort=nr_of_downloads&id=31
dataset_name = "credit-g"

# Configuracao de dados de saida
output_folder = "output"
output_dataset_file_path = "data/" + dataset_name + "/raw/"

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

dataset = fetch_openml(name=dataset_name, as_frame=True)

log.info("Dataset loaded!")
log.info("Dataset name: " + dataset_name)
log.info("Dataset description: \n" + str(dataset.DESCR))
log.info("Dataset features: " + str(dataset.details))

# construindo dataframe unico
data = pd.concat([dataset.data, dataset.target], axis=1)
describe_dataframe(data, log=log)

# incluindo coluna de data para desenvolvimento de funcionalidades
data["transaction_date"] = np.random.choice(
    pd.date_range("2023-01-01", "2023-12-31"), size=len(data)
)
data.index = pd.to_datetime(data["transaction_date"], format="%Y-%m-%d")
data["transaction_date_month"] = data.index.to_period("M").to_timestamp()
data.reset_index(drop=True, inplace=True)

# Criar identificador unico
data = create_unique_id(data=data, id_column_name="id", log=log)

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

save_dataframe_to_parquet(
    data, output_dataset_file_path + dataset_name + ".parquet", log=log
)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
