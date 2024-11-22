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
from xploreds.data_handler.file import load_dataframe_from_parquet
from xploreds.data_analysis.eda import (
    descriptive_analysis,
    trend_analysis,
    categorical_target_association_analysis,
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

# Analises exploratorias
exec_descritive_analysis = False

exec_trend_analysis = False
trend_analysis_date_ref = "transaction_date"
trend_analysis_date_trunc = "M"

exec_association_analysis = True

view_plots = True
save_plots = True
save_analysis = True

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

data = load_dataframe_from_parquet(file_path=input_dataset_file_path, log=log)

# ==================================================================================
# Regras de negócio
# ==================================================================================

if exec_descritive_analysis:
    log.title("Descriptive analysis")
    descriptive_analysis(
        data=data,
        numerical_variables=data.select_dtypes(include=[np.number]).columns,
        categorical_variables=data.select_dtypes(["category"]).columns,
        view_plots=view_plots,
        save_plots=save_plots,
        save_analysis=save_analysis,
        output_folder_path=log.log_path,
        prefix_label="eda_",
        log=log,
    )

if exec_trend_analysis:
    log.title("Trend analysis")
    trend_analysis(
        data=data,
        date_col_name=trend_analysis_date_ref,
        date_trunc_by=trend_analysis_date_trunc,
        numerical_variables=data.select_dtypes(include=[np.number]).columns,
        categorical_variables=data.select_dtypes(["category"]).columns,
        view_plots=view_plots,
        save_plots=save_plots,
        save_analysis=save_analysis,
        output_folder_path=log.log_path,
        prefix_label="trend_",
        log=log,
    )

if exec_association_analysis:
    log.title("Association analysis")
    categorical_target_association_analysis(
        data=data,
        target_col_name="class_bad",
        date_col_name=trend_analysis_date_ref,
        numerical_variables=["duration"],
        categorical_variables=["credit_history", "purpose"],
        view_plots=view_plots,
        save_plots=save_plots,
        save_analysis=save_analysis,
        output_folder_path=log.log_path,
        prefix_label="assoc_",
        log=log,
    )
# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
