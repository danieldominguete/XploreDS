"""
Xplore DS :: Feature Book for E-commerce Orders Dataset

"""

# Importando bibliotecas nativas
import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings

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
from xploreds.data_handler.dataframe import rename_columns, cast_columns_type_by_prefix
from xploreds.data_transformation.data_datetime_encoding import (
    build_relative_datetime_features,
    build_encoded_datetime_features,
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

# Configuracao de dados de entrada
input_reviews_file_path = "data/ecommerce/curated/olist_orders_curated_dataset.parquet"

# Configuracao de dados de saida
output_dataset_file_path = (
    "data/ecommerce/stage/olist_orders_feature_book_dataset.parquet"
)


# ==================================================================================
# Funcoes auxiliares


# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

data = load_dataframe_from_parquet(
    file_path=input_reviews_file_path,
    selected_columns=[
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    log=log,
)

# ==================================================================================
# Pré-processamento de dados
# ==================================================================================
log.title("Preprocessing datasets")

log.subtitle("Removing duplicates")

log.info("Removing duplicates from reviews dataset")
data = data.drop_duplicates(subset=["order_id"], keep="first")
log.info(f"Dataframe shape after removing duplicates: {data.shape[0]} rows")

log.subtitle("Rename columns")

data = rename_columns(
    data=data,
    columns_to_rename={
        "order_id": "cat_order_id",
        "customer_id": "cat_customer_id",
        "order_status": "cat_order_status",
        "order_purchase_timestamp": "tsp_order_purchase_timestamp",
        "order_approved_at": "tsp_order_approved_at",
        "order_delivered_carrier_date": "tsp_order_delivered_carrier_date",
        "order_delivered_customer_date": "tsp_order_delivered_customer_date",
        "order_estimated_delivery_date": "tsp_order_estimated_delivery_date",
    },
    log=log,
)

log.subtitle("Casting columns to appropriate types")
data = cast_columns_type_by_prefix(
    data=data,
    log=log,
)

# ==================================================================================
# Regras de negócio
# ==================================================================================
log.title("Applying business rules")

# Aplicando tempos relativos
reference_datetime_column = "tsp_order_purchase_timestamp"
for column in data.columns:
    if column.startswith("tsp_"):

        data = build_relative_datetime_features(
            data=data,
            variable_column_name=column,
            variable_reference_column_name=reference_datetime_column,
            time_reference="minutes",
            log=log,
        )

# Aplicando features de representação de tempo
for column in data.columns:
    if column.startswith("tsp_"):

        data = build_encoded_datetime_features(
            data=data, variable_column_name=column, log=log
        )

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

log.subtitle("Saving dataframe")
save_dataframe_to_parquet(
    data=data,
    file_path=output_dataset_file_path,
    log=log,
)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
