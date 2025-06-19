"""
Xplore DS :: Build Raw Data for Ecommerce Dataset
"""

# Importando bibliotecas nativas
import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[4]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xploreds.environment.environment import XploreDSLocalhost
from xploreds.environment.logging import XploreDSLogging
from xploreds.data_handler.file import (
    load_dataframe_from_csv,
    save_dataframe_to_parquet,
)
from xploreds.data_handler.dataframe import rename_columns, cast_columns_type_by_prefix

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


# Configuracao de dados de entrada
input_dataset_file_path_separator = ","
input_payments_file_path = "data/ecommerce/raw/olist_order_payments_dataset.csv"

# Configuracao de dados de saida
output_dataset_file_path = (
    "data/ecommerce/curated/olist_payments_curated_dataset.parquet"
)

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

payments_df = load_dataframe_from_csv(
    file_path=input_payments_file_path,
    separator=input_dataset_file_path_separator,
    log=log,
)

# ==================================================================================
# Pré-processamento de dados
# ==================================================================================

log.title("Removing duplicates from payments dataset")
payments_df = payments_df.drop_duplicates(
    subset=["order_id", "payment_sequential"], keep="first"
)
log.info(f"Dataframe shape after removing duplicates: {payments_df.shape[0]} rows")


# ==================================================================================
# Regras de negócio
# ==================================================================================

log.title("Applying business rules")


# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

log.subtitle("Casting columns to appropriate types")
cast_columns_type_by_prefix(
    data=reviews_df,
    log=log,
)

save_dataframe_to_parquet(
    data=payments_df,
    file_path=output_dataset_file_path,
    log=log,
)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
