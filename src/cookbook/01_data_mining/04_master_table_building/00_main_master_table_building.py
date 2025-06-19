"""
Xplore DS :: Preparing master data table
"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[4]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xploreds.environment.environment import XploreDSLocalhost
from xploreds.environment.logging import XploreDSLogging
from xploreds.data_handler.file import (
    load_dataframe_from_parquet,
    save_dataframe_to_parquet,
)

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

# ==================================================================================
# Parametrizacao do script
# ==================================================================================

log.title("Script setup")

# Configuracao de dados de entrada (arquivo de referencia "left join")
reference_dataset_file_path = "data/credit-g/curated/credit-g.parquet"

aditional_datasets_file_path = [
    "data/credit-g/stage/credit-g_numerical_scaling.parquet",
    "data/credit-g/stage/credit-g_categorical_encoding.parquet",
]

# Configuracao de dados de saida
output_dataset_file_path = "data/credit-g/processed/credit-g_master_table.parquet"

# Chave primaria de join de tabelas
primary_key = ["id"]

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

data = load_dataframe_from_parquet(file_path=reference_dataset_file_path, log=log)

# ==================================================================================
# Regras de negócio
# ==================================================================================

for dataset_file_path in aditional_datasets_file_path:

    dataset = load_dataframe_from_parquet(file_path=dataset_file_path, log=log)

    common_cols = [
        col for col in data.columns if col in dataset.columns and col not in primary_key
    ]

    # Drop duplicate columns from the dataset being merged
    log.info(f"Columns to drop: {common_cols}")
    dataset = dataset.drop(columns=common_cols)

    data = data.merge(dataset, on=primary_key, how="left")

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

save_dataframe_to_parquet(file_path=output_dataset_file_path, data=data, log=log)



# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
