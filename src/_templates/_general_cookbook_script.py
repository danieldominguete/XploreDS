"""
Xplore DS :: General cookbook script template
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
from xplore_ds.environment.environment import XploreDSLocalhost
from xplore_ds.environment.logging import XploreDSLogging

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

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

# ==================================================================================
# Regras de negócio
# ==================================================================================

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
