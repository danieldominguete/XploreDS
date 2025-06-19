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
)
from xploreds.data_handler.missing import (
    replace_missing_values_by_default_value,
    normalize_not_valid_values,
    replace_missing_values_by_statistics_value,
)
from xploreds.data_handler.outliers import (
    remove_unidimensional_outliers_by_zscore,
    remove_multidimensional_outliers_by_elliptic_envelope,
    remove_unidimensional_outliers_by_iqr,
    remove_unidimensional_outliers_by_winsorizing,
    remove_multidimensional_outliers_by_isolation_forest,
    replace_unidimensional_outliers_by_winsorizing,
)
from xploreds.data_transformation.data_scaling import (
    scaler_variable_fit_transform,
)
from xploreds.data_schemas.pre_processing_config import ScalingMethod

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

# Tratando missing values
data = replace_missing_values_by_default_value(
    data=data, column_names=["duration", "age"], replacement_value=0, log=log
)

data = replace_missing_values_by_statistics_value(
    data=data, column_names=["duration", "age"], replacement_value="median", log=log
)

# Tratando outliers unidimensionais
data = remove_unidimensional_outliers_by_zscore(
    data=data, column_names=["duration", "age"], zscore_threshold=3, log=log
)

data = remove_unidimensional_outliers_by_iqr(
    data=data, column_names=["duration", "age"], iqr_threshold=1.5, log=log
)

data = remove_unidimensional_outliers_by_winsorizing(
    data=data,
    column_names=["duration", "age"],
    max_percentile_threshold=0.99,
    min_percentile_threshold=0.01,
    log=log,
)

data = replace_unidimensional_outliers_by_winsorizing(
    data=data,
    column_names=["duration", "age"],
    max_percentile_threshold=0.99,
    min_percentile_threshold=0.01,
    log=log,
)

# Tratando outliers multidimensionais
data = remove_multidimensional_outliers_by_elliptic_envelope(
    data=data,
    column_names=["duration", "age"],
    contamination=0.1,
    log=log,
)

data = remove_multidimensional_outliers_by_isolation_forest(
    data=data,
    column_names=["duration", "age"],
    contamination=0.1,
    log=log,
)

# Normalizando valores de colunas
data, _ = scaler_variable_fit_transform(
    data=data,
    variable_column_name="duration",
    scale_method=ScalingMethod.mean_std_scaler,
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
