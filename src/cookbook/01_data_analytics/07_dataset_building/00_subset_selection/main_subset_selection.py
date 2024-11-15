"""
Xplore DS :: Subset selection script template
"""

# Importando bibliotecas nativas
import sys, os
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
from xploreds.data_handler.subsets import (
    create_train_test_data_subsets,
    generate_features_config_default,
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
input_dataset_file_path = "data/credit-g/processed/credit-g_master_table.parquet"

# Selecao dos subsets
proportion_test_samples = 0.1
shuffle = False
random_state = 100

# Configuracao de dados de saida
output_dataset_train_file_path = "data/credit-g/processed/credit-g_train.parquet"
output_dataset_test_file_path = "data/credit-g/processed/credit-g_test.parquet"

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

data = load_dataframe_from_parquet(file_path=input_dataset_file_path, log=log)

# ==================================================================================
# Regras de negócio
# ==================================================================================

# Realizando o split dos datasets
data_train, data_test = create_train_test_data_subsets(
    data=data,
    proportion_test_samples=proportion_test_samples,
    shuffle=shuffle,
    random_state=random_state,
    log=log,
)

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

save_dataframe_to_parquet(
    data=data_train,
    file_path=output_dataset_train_file_path,
    log=log,
)

save_dataframe_to_parquet(
    data=data_test, file_path=output_dataset_test_file_path, log=log
)

generate_features_config_default(
    data=data_train,
    file_path="data/credit-g/processed/credit-g_features_config_template.json",
    log=log,
)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
