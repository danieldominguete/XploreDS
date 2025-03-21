"""
Xplore DS :: Time series aggregation features cookbook script template
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
from xploreds.data_transformation.data_aggregation import (
    generate_features_by_entity_aggregation_and_timing_references_for_numerical_variables,
)
from xploreds.data_handler.date_time import create_past_datetime_mask_from_reference


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
output_feature_book_file_path = "data/credit-g/stage/credit-g_time_series_book.parquet"

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

df_raw = df_raw.drop_duplicates(
    subset=["customer_name", "order_date"],
    keep="first",  # 'first', 'last', or 'False' to drop all duplicates
)

# truncar data para unidade de referencia temporal
df_raw["order_date"] = df_raw["order_date"].dt.to_period("D").dt.to_timestamp()
df_raw["dt_predict"] = pd.to_datetime(df_raw["order_date"].max())

# ==================================================================================
# Regras de negócio
# ==================================================================================

df_book = generate_features_by_entity_aggregation_and_timing_references_for_numerical_variables(
    data=df_raw,
    id_entity_reference_column_name="customer_name",
    feature_datetime_reference_column_name="dt_predict",
    datetime_pre_summarization_step_unit="M",
    raw_data_datetime_reference_column_name="order_date",
    raw_data_past_steps_window_from_reference=3,
    numerical_variables_columns_names=["total"],
    log=log,
)

# # criação da mascara de slots temporais
# df_mask = create_past_datetime_mask_from_reference(
#     data=df_raw,
#     id_entity_column_name="customer_name",
#     t0_reference_column_name="order_date",
#     time_step_unit="months",
#     time_step_amount=3,
#     log=log,
# )

# df_mask = df_mask.sort_values(by=["customer_name", "order_date", "step_id"])

# agregar eventos na mesma unidade temporal


# agregar features na janela de tempo amostral


# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

# save_dataframe_to_parquet(
#     data=df_book,
#     file_path=output_feature_book_file_path,
#     log=log,
# )

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
