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
from xplore_ds.environment.environment import XploreDSLocalhost
from xplore_ds.environment.logging import XploreDSLogging
from xplore_ds.data_handler.file import (
    load_dataframe_from_csv,
    save_dataframe_to_parquet,
)
from sklearn.model_selection import train_test_split


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
input_dataset_file_path = (
    "data/projects/raw/tabular_data/wine_quality/winequality-red.csv"
)
input_dataset_file_path_separator = ","

# Selecao dos subsets
proportion_test_samples = 0.1
shuffle = False
random_state = 100

# Configuracao de dados de saida
output_folder = "output"
output_dataset_train_file_path = (
    "data/projects/stage/wine_quality/wine_quality_train.parquet"
)
output_dataset_test_file_path = (
    "data/projects/stage/wine_quality/wine_quality_test.parquet"
)


# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

data = load_dataframe_from_csv(
    filepath=input_dataset_file_path,
    separator=input_dataset_file_path_separator,
    log=log,
)

# ==================================================================================
# Regras de negócio
# ==================================================================================

# Realizando o split dos datasets
data_train, data_test = train_test_split(
    data,
    test_size=proportion_test_samples,
    shuffle=shuffle,
    random_state=random_state,
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

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
