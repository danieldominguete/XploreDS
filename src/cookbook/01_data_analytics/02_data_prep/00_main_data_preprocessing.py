"""
Xplore DS :: Main dataset preprocessing script template
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
from xploreds.data_handler.dataframe import (
    rename_columns,
    normalize_not_valid_values,
    create_unique_id,
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
input_dataset_file_path = "data/credit-g/raw/credit-g.parquet"

# Configuracao de dados de saida
output_dataset_file_path = "data/credit-g/curated/credit-g.parquet"

# Parametros de negocio
columns_to_rename = {}

# Criacao de chave unica
id_column_name = "id"

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

data = load_dataframe_from_parquet(file_path=input_dataset_file_path, log=log)

# ==================================================================================
# Regras de negócio
# ==================================================================================
log.title("Preprocessing dataset")

# Rename columns
data = rename_columns(data=data, columns_to_rename=columns_to_rename, log=log)

# Normalizar not valid values
data = normalize_not_valid_values(data=data, log=log)

# Criar identificador unico
data = create_unique_id(data=data, id_column_name=id_column_name, log=log)

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

save_dataframe_to_parquet(data=data, file_path=output_dataset_file_path, log=log)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
