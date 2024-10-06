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
from xplore_ds.data_handler.file import load_csv_database


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

# ==================================================================================
# Funcoes do script
# ==================================================================================

# ==================================================================================
# Parametrizacao do script
# ==================================================================================

log.title("Parametrizacao do script")

# Configuracao de dados de entrada
input_dataset_filepath = (
    "data/projects/raw/tabular_data/wine_quality/winequality-red.csv"
)
input_dataset_filepath_separator = ","

# Configuracao de dados de saida
output_folder = "output"
output_dataset_train_filepath = (
    "data/projects/stage/wine_quality/wine_quality_train.csv"
)
output_dataset_test_filepath = "data/projects/stagewine_quality/wine_quality_test.csv"


# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Carregando base de dados")

data = load_csv_database(
    filepath=input_dataset_filepath, separator=input_dataset_filepath_separator, log=log
)

# ==================================================================================
# Regras de negócio
# ==================================================================================


# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
