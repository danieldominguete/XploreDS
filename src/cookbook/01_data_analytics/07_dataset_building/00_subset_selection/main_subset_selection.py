"""
Xplore DS :: Subset selection script template
"""

# Importando bibliotecas nativas
import sys, os
from pathlib import Path
from dotenv import load_dotenv
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
from xploreds.data_handler.subsets import (
    create_train_test_data_subsets,
    generate_features_config_default,
    check_drift_subsets,
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
proportion_out_of_samples = 0.1
shuffle = False
random_state = 100

date_reference_column_name = "transaction_date"
out_of_time_date_min = "2024-12-01"

# Configuracao de dados de saida
output_dataset_train_file_path = "data/credit-g/processed/credit-g_train.parquet"
output_dataset_out_of_sample_file_path = (
    "data/credit-g/processed/credit-g_out_of_sample.parquet"
)
output_dataset_out_of_time_file_path = (
    "data/credit-g/processed/credit-g_out_of_time.parquet"
)


# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

data = load_dataframe_from_parquet(file_path=input_dataset_file_path, log=log)

# ==================================================================================
# Regras de negócio
# ==================================================================================

# Realizando corte do out_of_time
data_oot = data[data[date_reference_column_name] >= out_of_time_date_min]
data = data[data[date_reference_column_name] < out_of_time_date_min]

# Realizando o split dos datasets
data_train, data_oos = create_train_test_data_subsets(
    data=data,
    proportion_test_samples=proportion_out_of_samples,
    shuffle=shuffle,
    random_state=random_state,
    log=log,
)

log.info("Train dataset shape: {}".format(data_train.shape))
log.info("Out of sample dataset shape: {}".format(data_oos.shape))
log.info("Out of time dataset shape: {}".format(data_oot.shape))

# ==================================================================================
# Analisando consistencia dos conjuntos
# ==================================================================================

log.title("Analyzing subsets")


check_drift_subsets(
    data_train=data_train,
    data_oos=data_oos,
    data_oot=data_oot,
    log=log,
    view_plots=True,
    save_plots=True,
    output_folder_path=log.log_path,
    prefix_label="subsets",
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
    data=data_oos, file_path=output_dataset_out_of_sample_file_path, log=log
)

save_dataframe_to_parquet(
    data=data_oot, file_path=output_dataset_out_of_time_file_path, log=log
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
