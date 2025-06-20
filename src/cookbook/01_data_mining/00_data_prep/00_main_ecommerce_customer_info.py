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
input_orders_file_path = "data/ecommerce/raw/olist_orders_dataset.csv"
input_customers_file_path = "data/ecommerce/raw/olist_customers_dataset.csv"
input_zipcodes_file_path = "data/ecommerce/raw/olist_geolocation_dataset.csv"

# Configuracao de dados de saida
output_dataset_file_path = (
    "data/ecommerce/curated/olist_customer_curated_dataset.parquet"
)

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

orders = load_dataframe_from_csv(
    file_path=input_orders_file_path,
    separator=input_dataset_file_path_separator,
    log=log,
)

orders = rename_columns(
    data=orders,
    columns_to_rename={
        "order_id": "cat_order_id",
        "customer_id": "cat_customer_id",
        "order_status": "cat_order_status",
        "order_purchase_timestamp": "ts_order_purchase_timestamp",
        "order_approved_at": "ts_order_approved_at",
        "order_delivered_carrier_date": "ts_order_delivered_carrier_date",
        "order_delivered_customer_date": "ts_order_delivered_customer_date",
        "order_estimated_delivery_date": "dt_order_estimated_delivery_date",
    },
    log=log,
)

customer = load_dataframe_from_csv(
    file_path=input_customers_file_path,
    separator=input_dataset_file_path_separator,
    log=log,
)
customer = rename_columns(
    data=customer,
    columns_to_rename={
        "customer_id": "cat_customer_id",
        "customer_unique_id": "cat_customer_unique_id",
        "customer_zip_code_prefix": "cat_customer_zip_code_prefix",
        "customer_city": "cat_customer_city",
        "customer_state": "cat_customer_state",
    },
    log=log,
)

zipcodes = load_dataframe_from_csv(
    file_path=input_zipcodes_file_path,
    separator=input_dataset_file_path_separator,
    log=log,
)

zipcodes = rename_columns(
    data=zipcodes,
    columns_to_rename={
        "geolocation_zip_code_prefix": "cat_geolocation_zip_code_prefix",
        "geolocation_lat": "num_geolocation_lat",
        "geolocation_lng": "num_geolocation_lng",
        "geolocation_city": "cat_geolocation_city",
        "geolocation_state": "cat_geolocation_state",
    },
    log=log,
)

# ==================================================================================
# Pré-processamento de dados
# ==================================================================================

log.title("Removing duplicates from geolocation dataset")
zipcodes = zipcodes.drop_duplicates(subset=["cat_geolocation_zip_code_prefix"])
log.info(f"Dataframe shape after removing duplicates: {zipcodes.shape[0]} rows")
zipcodes = zipcodes.add_suffix("_customer")

# ==================================================================================
# Regras de negócio
# ==================================================================================

log.title("Applying business rules")

log.info("Merging orders with customers...")
data = pd.merge(
    orders,
    customer,
    how="left",
    left_on="cat_customer_id",
    right_on="cat_customer_id",
    validate="one_to_one",
)
log.info(f"Dataframe shape after merging: {data.shape}")

log.info("Merging with geolocation...")
data = pd.merge(
    data,
    zipcodes,
    how="left",
    left_on="cat_customer_zip_code_prefix",
    right_on="cat_geolocation_zip_code_prefix_customer",
    validate="many_to_one",
)
data = data.drop(columns=["cat_geolocation_zip_code_prefix_customer"])
log.info(f"Dataframe shape after merging: {data.shape}")

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

log.subtitle("Casting columns to appropriate types")
data = cast_columns_type_by_prefix(
    data=data,
    log=log,
)

log.subtitle("Saving dataframe to file")
save_dataframe_to_parquet(
    data=data,
    file_path=output_dataset_file_path,
    log=log,
)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
