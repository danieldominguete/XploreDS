"""
Xplore DS :: Main categorical encoding script template
"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv


# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[6]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xploreds.environment.environment import XploreDSLocalhost
from xploreds.environment.logging import XploreDSLogging
from xploreds.data_handler.file import (
    load_dataframe_from_parquet,
    save_dataframe_to_parquet,
)
from xploreds.data_schemas.model_io_config import (
    VariableConfig,
    EncodingMethod,
)
from xploreds.data_transformation.data_categorical_encoding import (
    encoder_variable_fit_transform,
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
input_dataset_file_path = "data/credit-g/curated/credit-g.parquet"

# Configuracao de dados de saida
output_dataset_file_path = "data/credit-g/stage/credit-g_categorical_encoding.parquet"

# Configuracao das variaveis a serem processadas
categorical_variables_config = [
    VariableConfig(
        name="checking_status", encoding_method=EncodingMethod.one_hot_encoder
    ),
    VariableConfig(
        name="credit_history", encoding_method=EncodingMethod.one_hot_encoder
    ),
    VariableConfig(name="purpose", encoding_method=EncodingMethod.one_hot_encoder),
    VariableConfig(
        name="savings_status", encoding_method=EncodingMethod.one_hot_encoder
    ),
    VariableConfig(name="employment", encoding_method=EncodingMethod.one_hot_encoder),
    VariableConfig(
        name="personal_status", encoding_method=EncodingMethod.one_hot_encoder
    ),
    VariableConfig(
        name="other_parties", encoding_method=EncodingMethod.one_hot_encoder
    ),
    VariableConfig(
        name="property_magnitude", encoding_method=EncodingMethod.one_hot_encoder
    ),
    VariableConfig(
        name="other_payment_plans", encoding_method=EncodingMethod.one_hot_encoder
    ),
    VariableConfig(name="housing", encoding_method=EncodingMethod.one_hot_encoder),
    VariableConfig(name="job", encoding_method=EncodingMethod.one_hot_encoder),
    VariableConfig(
        name="own_telephone", encoding_method=EncodingMethod.one_hot_encoder
    ),
    VariableConfig(
        name="foreign_worker", encoding_method=EncodingMethod.one_hot_encoder
    ),
    VariableConfig(name="class", encoding_method=EncodingMethod.one_hot_encoder),
]

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

data = load_dataframe_from_parquet(file_path=input_dataset_file_path, log=log)

# ==================================================================================
# Regras de negócio
# ==================================================================================
log.title("Encoding categorical variables")

for var in categorical_variables_config:

    data, encoded_variables = encoder_variable_fit_transform(
        data=data,
        variable_column_name=var.name,
        encode_method=var.encoding_method,
        log=log,
    )


# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

save_dataframe_to_parquet(data=data, file_path=output_dataset_file_path, log=log)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
