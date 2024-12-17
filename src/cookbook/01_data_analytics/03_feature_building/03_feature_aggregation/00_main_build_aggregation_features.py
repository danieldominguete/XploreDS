"""
Xplore DS :: Feature aggregation cookbook script template
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
output_dataset_train_file_path = (
    "data/credit-g/stage/credit-g_aggregation_features.parquet"
)

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

# ==================================================================================
# Regras de negócio
# ==================================================================================

# agregacao por variaveis numericas associadas a uma chave primaria sem filtro temporal

log.title("Simple Numerical Aggregation")

df_raw = pd.merge(
    df_orders,
    df_order_products,
    left_on="order_id",
    right_on="order_id",
    how="left",
)

df_book = generate_primitive_numerical_features(
    data=df_raw,
    id_data_entity_column_name="customer_name",
    feature_column_name="total",
    log=log,
)

# agregacao por variaveis categoricas associadas a uma chave primaria sem filtro temporal

# agregacao por variaveis binarias associadas a uma chave primaria sem filtro temporal

# agregacao por variaveis textuais associadas a uma chave primaria sem filtro temporal

# agregacao por variaveis numericas sob uma janela de tempo (series temporais)

# agregacao por variaveis categoricas sob uma janela de tempo (series temporais)

# agregacao por variaveis binarias sob uma janela de tempo (series temporais)

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
