"""
Xplore DS :: Simple aggregation features cookbook script template
"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import featuretools as ft
import ssl
import pandas as pd

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[5]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xploreds.environment.environment import XploreDSLocalhost
from xploreds.environment.logging import XploreDSLogging
from xploreds.data_handler.file import (
    load_dataframe_from_parquet,
    save_dataframe_to_parquet,
)
from xploreds.variables.variables_aggregation import (
    generate_primitive_numerical_features,
    generate_primitive_categorical_features,
    generate_custom_numerical_features,
    generate_primitive_datetime_features,
    generate_custom_datetime_features,
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


# Configuracao de dados de saida
output_feature_book_file_path = "data/credit-g/stage/credit-g_aggregation_book.parquet"

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

# Create a default SSL context with certificate verification disabled
ssl._create_default_https_context = ssl._create_unverified_context

# Downloading data
mock_data = ft.demo.load_retail()
df_orders = mock_data["orders"]
df_products = mock_data["products"]
df_customers = mock_data["customers"]
df_order_products = mock_data["order_products"]

df_raw = pd.merge(
    df_orders,
    df_order_products,
    left_on="order_id",
    right_on="order_id",
    how="left",
)

# ==================================================================================
# Regras de negócio
# ==================================================================================

# agregacao por variaveis numericas associadas a uma chave primaria sem filtro temporal

log.title("Numerical Features Simple Aggregation")

df_book_1 = generate_primitive_numerical_features(
    data=df_raw,
    id_data_entity_column_name="customer_name",
    numerical_features_columns_names=["total", "quantity"],
    log=log,
)

df_book_2 = generate_custom_numerical_features(
    data=df_raw,
    id_data_entity_column_name="customer_name",
    numerical_features_columns_names=["total", "quantity"],
    log=log,
)

log.title("Categorical Features Simple Aggregation")

df_book_3 = generate_primitive_categorical_features(
    data=df_raw,
    id_data_entity_column_name="customer_name",
    categorical_features_columns_names=["product_id"],
    log=log,
)

log.title("Datetime Features Simple Aggregation")

df_book_4 = generate_primitive_datetime_features(
    data=df_raw,
    id_data_entity_column_name="customer_name",
    datetime_columns=["order_date"],
    log=log,
)

df_book_5 = generate_custom_datetime_features(
    data=df_raw,
    id_data_entity_column_name="customer_name",
    datetime_columns=["order_date"],
    log=log,
)

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

df_book = pd.merge(
    df_book,
    df_book_2,
    left_on="customer_name",
    right_on="customer_name",
    how="left",
)

df_book = pd.merge(
    df_book,
    df_book_3,
    left_on="customer_name",
    right_on="customer_name",
    how="left",
)

save_dataframe_to_parquet(
    data=df_book,
    file_path=output_feature_book_file_path,
    log=log,
)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
