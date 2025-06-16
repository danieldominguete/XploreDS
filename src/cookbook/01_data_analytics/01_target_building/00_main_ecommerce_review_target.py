"""
Xplore DS :: Build Target for Ecommerce Dataset

Classificacao Binaria = "Review Positiva" ou "Review Negativa"
Classificacao Multiclasse = 0 a 5
Predicao de Regressao = "Nota da Review"

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
input_reviews_file_path = "data/ecommerce/raw/olist_order_reviews_dataset.csv"

# Configuracao de dados de saida
output_dataset_file_path = "data/ecommerce/stage/olist_review_target_dataset.parquet"

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

reviews_df = load_dataframe_from_csv(
    filepath=input_reviews_file_path,
    separator=input_dataset_file_path_separator,
    selected_columns=[
        "order_id",
        "review_score",
    ],
    log=log,
)

# ==================================================================================
# Pré-processamento de dados
# ==================================================================================
log.title("Preprocessing datasets")

log.subtitle("Removing duplicates")

log.info("Removing duplicates from reviews dataset")
reviews_df = reviews_df.drop_duplicates(subset=["order_id"], keep="first")
log.info(f"Dataframe shape after removing duplicates: {reviews_df.shape[0]} rows")

log.subtitle("Rename columns")

reviews_df = rename_columns(
    data=reviews_df,
    columns_to_rename={
        "order_id": "txt_order_id",
        "review_score": "num_review_score",
    },
    log=log,
)

# ==================================================================================
# Regras de negócio
# ==================================================================================

log.title("Applying business rules")

reviews_df["txt_review_binary"] = reviews_df["num_review_score"].apply(
    lambda x: "Review Positiva" if x >= 4 else "Review Negativa"
)

reviews_df["txt_review_multiclass"] = reviews_df["num_review_score"].apply(
    lambda x: (
        "Review 5"
        if x == 5
        else (
            "Review 4"
            if x == 4
            else (
                "Review 3"
                if x == 3
                else (
                    "Review 2"
                    if x == 2
                    else "Review 1" if x == 1 else "Review Desconhecida"
                )
            )
        )
    )
)

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

log.subtitle("Casting columns to appropriate types")
cast_columns_type_by_prefix(
    data=reviews_df,
    log=log,
)

log.subtitle("Saving dataframe")
save_dataframe_to_parquet(
    data=reviews_df,
    file_path=output_dataset_file_path,
    log=log,
)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
