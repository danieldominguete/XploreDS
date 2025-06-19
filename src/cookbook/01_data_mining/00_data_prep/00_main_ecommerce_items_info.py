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
input_items_file_path = "data/ecommerce/raw/olist_order_items_dataset.csv"
input_products_file_path = "data/ecommerce/raw/olist_products_dataset.csv"
input_sellers_file_path = "data/ecommerce/raw/olist_sellers_dataset.csv"
input_zipcodes_file_path = "data/ecommerce/raw/olist_geolocation_dataset.csv"

# Configuracao de dados de saida
output_dataset_file_path = "data/ecommerce/curated/olist_items_curated_dataset.parquet"

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

items_df = load_dataframe_from_csv(
    file_path=input_items_file_path,
    separator=input_dataset_file_path_separator,
    log=log,
)

products_df = load_dataframe_from_csv(
    file_path=input_products_file_path,
    separator=input_dataset_file_path_separator,
    log=log,
)

sellers_df = load_dataframe_from_csv(
    file_path=input_sellers_file_path,
    separator=input_dataset_file_path_separator,
    log=log,
)

zipcodes_df = load_dataframe_from_csv(
    file_path=input_zipcodes_file_path,
    separator=input_dataset_file_path_separator,
    log=log,
)

# ==================================================================================
# Pré-processamento de dados
# ==================================================================================

log.title("Removing duplicates from items dataset")
items_df = items_df.drop_duplicates(subset=["order_id", "order_item_id"], keep="first")
log.info(f"Dataframe shape after removing duplicates: {items_df.shape[0]} rows")

log.title("Removing duplicates from products dataset")
products_df = products_df.drop_duplicates(subset=["product_id"], keep="first")
log.info(f"Dataframe shape after removing duplicates: {products_df.shape[0]} rows")

log.title("Removing duplicates from products dataset")
sellers_df = sellers_df.drop_duplicates(subset=["seller_id"], keep="first")
log.info(f"Dataframe shape after removing duplicates: {sellers_df.shape[0]} rows")

log.title("Removing duplicates from geolocation dataset")
zipcodes_df = zipcodes_df.drop_duplicates(subset=["geolocation_zip_code_prefix"])
log.info(f"Dataframe shape after removing duplicates: {zipcodes_df.shape[0]} rows")
zipcodes_df = zipcodes_df.add_suffix("_seller")

# ==================================================================================
# Regras de negócio
# ==================================================================================

log.title("Applying business rules")

log.info("Merging items with products...")
data_df = pd.merge(
    items_df,
    products_df,
    how="left",
    left_on="product_id",
    right_on="product_id",
    validate="many_to_one",
)
log.info(f"Dataframe shape after merging: {data_df.shape}")

log.info("Merging items with sellers...")
data_df = pd.merge(
    data_df,
    sellers_df,
    how="left",
    left_on="seller_id",
    right_on="seller_id",
    validate="many_to_one",
)
log.info(f"Dataframe shape after merging: {data_df.shape}")

log.info("Merging with geolocation...")
data_df = pd.merge(
    data_df,
    zipcodes_df,
    how="left",
    left_on="seller_zip_code_prefix",
    right_on="geolocation_zip_code_prefix_seller",
    validate="many_to_one",
)
log.info(f"Dataframe shape after merging: {data_df.shape}")

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
    data=data_df,
    file_path=output_dataset_file_path,
    log=log,
)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
