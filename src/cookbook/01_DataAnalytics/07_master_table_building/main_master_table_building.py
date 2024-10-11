"""
Xplore DS :: Preparing master data table
"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from sklearn.model_selection import train_test_split

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[4]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xplore_ds.environment.environment import XploreDSLocalhost
from xplore_ds.environment.logging import XploreDSLogging
from xplore_ds.data_handler.file import (
    load_dataframe_from_csv,
    save_dataframe_to_parquet,
)
from xplore_ds.data_schemas.dataset_config import (
    VariableConfig,
    EncodingMethod,
    DatasetConfig,
)

from xplore_ds.variables.variables_encoding import encoder_variable_fit_transform

# **********************************************************************************
# Parametrizacao do script
# **********************************************************************************

# ==================================================================================
# Setup de ambiente
# ==================================================================================

script_name = os.path.basename(__file__)

# Variaveis de ambiente
load_dotenv()

# Criando estrutura de execucao local
env = XploreDSLocalhost(run_folder=project_folder)

# Criando estrutura de logs
log = XploreDSLogging(project_root=project_folder, script_name=script_name)
log.init_run()

# ==================================================================================
# Parametrizacao de execucao
# ==================================================================================

log.title("Script execution setup")

# ----------------------------------------------------------------------------------
# Seeds
random_state = 100

# ----------------------------------------------------------------------------------
# Configuracao de dados de entrada

input_dataset_file_path = (
    "data/projects/raw/tabular_data/wine_quality/winequality-red.csv"
)

# ----------------------------------------------------------------------------------
# Configuracao de variaveis

volatile_acidity = VariableConfig(name="volatile acidity")
citric_acid = VariableConfig(name="citric acid")
residual_sugar = VariableConfig(name="residual sugar")
chlorides = VariableConfig(
    name="chlorides",
)
free_sulfur_dioxide = VariableConfig(name="free sulfur dioxide")
total_sulfur_dioxide = VariableConfig(name="total sulfur dioxide")
density = VariableConfig(name="density")
pH = VariableConfig(name="pH")
pH_label = VariableConfig(
    name="pH_label",
    encoding_method=EncodingMethod.one_hot_encoder,
)
sulphates = VariableConfig(name="sulphates")
alcohol = VariableConfig(name="alcohol")
fixed_acidity = VariableConfig(name="fixed acidity")
target = VariableConfig(name="quality")
target_label = VariableConfig(
    name="quality_label", encoding_method=EncodingMethod.one_hot_encoder
)

# ----------------------------------------------------------------------------------
# Configurando a base de conhecimento "ground thruth" para tunning do modelo

dataset_config = DatasetConfig(
    variables=[
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        pH_label,
        sulphates,
        alcohol,
        fixed_acidity,
        target,
        target_label,
    ]
)

# ----------------------------------------------------------------------------------

# Selecao dos subsets
proportion_test_samples = 0.1
shuffle = False
random_state = 100

# ----------------------------------------------------------------------------------
# Configuracao de artefatos de saida

results_folder = log.log_path

output_dataset_file_path = results_folder + "winequality-red-processed.parquet"

output_dataset_train_file_path = (
    results_folder + "winequality-red-processed-train.parquet"
)
output_dataset_test_file_path = (
    results_folder + "winequality-red-processed-test.parquet"
)

# **********************************************************************************
# Execucao do script
# **********************************************************************************

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

data = load_dataframe_from_csv(filepath=input_dataset_file_path, separator=",", log=log)

# marcacao para testes de variaveis categoricas
data["pH_label"] = np.where(data["pH"] <= 3, "acid", "base")
data["quality_label"] = np.where(data["quality"] <= 5, "bad", "good")

# ==================================================================================
# Regras de negócio
# ==================================================================================

log.title("Encoding variables")

# ----------------------------------------------------------------------------------
# Criando encoding de variaveis nao numericas

log.info("Encoding variables...")

for variable in dataset_config.variables:

    data, encoded_variables = encoder_variable_fit_transform(
        data=data,
        variable_column_name=variable.name,
        encode_method=variable.encoding_method,
        log=log,
    )

# ==================================================================================
# Separando datasets
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
    data=data,
    file_path=output_dataset_file_path,
    log=log,
)

save_dataframe_to_parquet(
    data=data_train,
    file_path=output_dataset_train_file_path,
    log=log,
)

save_dataframe_to_parquet(
    data=data_test,
    file_path=output_dataset_test_file_path,
    log=log,
)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
