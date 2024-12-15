"""
Xplore DS :: Main numerical scaling script template
"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv


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
from xploreds.data_schemas.model_io_config import (
    VariableConfig,
    ScalingMethod,
)
from xploreds.variables.variables_scaling import scaler_variable_fit_transform

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
output_dataset_file_path = "data/credit-g/stage/credit-g_numerical_scaling.parquet"

# Configuracao das variaveis a serem processadas
categorical_variables_config = [
    VariableConfig(name="duration", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="credit_amount", scaling_method=ScalingMethod.mean_std_scaler),
    VariableConfig(
        name="installment_commitment", scaling_method=ScalingMethod.min_max_scaler
    ),
    VariableConfig(name="num_dependents", scaling_method=ScalingMethod.min_max_scaler),
]

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

data = load_dataframe_from_parquet(file_path=input_dataset_file_path, log=log)

# ==================================================================================
# Regras de negócio
# ==================================================================================
log.title("Scaling numerical variables")

for var in categorical_variables_config:

    data, scaled_variables = scaler_variable_fit_transform(
        data=data,
        variable_column_name=var.name,
        scale_method=var.scaling_method,
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
