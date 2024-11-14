"""
Xplore DS :: General cookbook script template
"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np


# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[4]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xploreds.environment.environment import XploreDSLocalhost
from xploreds.environment.logging import XploreDSLogging
from xploreds.data_handler.file import (
    load_dataframe_from_parquet,
)
from xploreds.data_visualization.data_viz_plotly import (
    plot_histogram,
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
view_plots = True
save_plots = True

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")
data = load_dataframe_from_parquet(file_path=input_dataset_file_path, log=log)

# ==================================================================================
# Regras de negócio
# ==================================================================================

# listando variaveis por natureza
categorical_columns = data.select_dtypes(["category"]).columns
numerical_columns = data.select_dtypes(include=[np.number]).columns

# plotando visualizacoes das variaveis categoricas
for v in categorical_columns:

    log.info("Ploting visualization of " + v + "...")
    plot_histogram(
        data=data,
        x_col_name=v,
        title="Histogram of " + v,
        view_chart=view_plots,
        save_chart=save_plots,
        file_path_image=log.log_path + "/histogram_" + v + ".png",
    )

# plotando visualizacoes das variaveis numericas
for v in numerical_columns:

    log.info("Ploting visualization of " + v + "...")
    plot_histogram(
        data=data,
        x_col_name=v,
        title="Histogram of " + v,
        view_chart=view_plots,
        save_chart=save_plots,
        file_path_image=log.log_path + "/histogram_" + v + ".png",
    )

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
