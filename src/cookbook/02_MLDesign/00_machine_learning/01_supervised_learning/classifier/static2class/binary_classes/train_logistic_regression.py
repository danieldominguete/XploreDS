"""
Xplore DS :: Training Logistic Regression Model
"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[8]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xplore_ds.environment.environment import XploreDSLocalhost
from xplore_ds.environment.logging import XploreDSLogging
from xplore_ds.data_handler.file import (
    load_dataframe_from_parquet,
    save_dataframe_to_parquet,
)
from xplore_ds.models.logistic_regression import XLogisticRegression
from xplore_ds.data_schemas.logistic_regression_config import (
    LogisticRegressionConfig,
    LogisticRegressionHyperparameters,
    Topology,
    FitAlgorithm,
)
from xplore_ds.data_schemas.model_io_config import (
    ModelIOConfig,
    VariableConfig,
    ScalingMethod,
    ApplicationType,
)

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

input_dataset_train_file_path = (
    "data/projects/stage/wine_quality/winequality-red-processed.parquet"
)
input_dataset_test_file_path = (
    "data/projects/stage/wine_quality/winequality-red-processed.parquet"
)

# ----------------------------------------------------------------------------------
# Configuracao de features e target

fixed_acidity = VariableConfig(
    name="fixed acidity", scaling_method=ScalingMethod.none_scaler
)
volatile_acidity = VariableConfig(
    name="volatile acidity", scaling_method=ScalingMethod.min_max_scaler
)
citric_acid = VariableConfig(
    name="citric acid", scaling_method=ScalingMethod.mean_std_scaler
)
residual_sugar = VariableConfig(
    name="residual sugar", scaling_method=ScalingMethod.none_scaler
)
chlorides = VariableConfig(name="chlorides", scaling_method=ScalingMethod.none_scaler)
free_sulfur_dioxide = VariableConfig(
    name="free sulfur dioxide", scaling_method=ScalingMethod.none_scaler
)
total_sulfur_dioxide = VariableConfig(
    name="total sulfur dioxide", scaling_method=ScalingMethod.none_scaler
)
density = VariableConfig(name="density", scaling_method=ScalingMethod.none_scaler)
pH_label_acid = VariableConfig(
    name="pH_label_acid",
    scaling_method=ScalingMethod.none_scaler,
)
sulphates = VariableConfig(name="sulphates", scaling_method=ScalingMethod.none_scaler)
alcohol = VariableConfig(name="alcohol", scaling_method=ScalingMethod.none_scaler)

quality_label_bad = VariableConfig(name="quality_label_bad")

# ----------------------------------------------------------------------------------
# Configurando a base de conhecimento "ground thruth" para tunning do modelo

model_io_config = ModelIOConfig(
    application_type=ApplicationType.binary_classification,
    features=[
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH_label_acid,
        sulphates,
        alcohol,
        fixed_acidity,
    ],
    target=[quality_label_bad],
)

# ----------------------------------------------------------------------------------
# Setup do modelo

model_config = LogisticRegressionConfig(
    set_intersection_with_zero=False, topology=Topology.logit
)

# ----------------------------------------------------------------------------------
# Hiperparametros

tunning_config = LogisticRegressionHyperparameters(
    fit_algorithm=FitAlgorithm.maximum_likelihood,
)

# ----------------------------------------------------------------------------------
# Configuracao de artefatos de saida

results_folder = log.log_path

output_dataset_train_predict_file_path = (
    results_folder + "data/wine_quality_train_classification_predict.parquet"
)
output_dataset_test_predict_file_path = (
    results_folder + "data/wine_quality_test_classification_predict.parquet"
)
output_model_file_path = (
    results_folder + "models/wine_quality_logistic_regression.joblib"
)

view_charts = True
save_charts = True

# **********************************************************************************
# Execucao do script
# **********************************************************************************

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

data_train = load_dataframe_from_parquet(
    file_path=input_dataset_train_file_path, log=log
)

# ==================================================================================
# Regras de negócio
# ==================================================================================

log.title("Training model")

# ----------------------------------------------------------------------------------
# Criando topologia do modelo

log.info("Creating model topology...")

model = XLogisticRegression(
    model_io_config=model_io_config,
    model_config=model_config,
    tunning_config=tunning_config,
    random_state=random_state,
    log=log,
)

# ----------------------------------------------------------------------------------
# Realizando do tunning do modelo

log.title("Training model")

model.fit(data=data_train)

# ----------------------------------------------------------------------------------
# Apresentando resumo do tunning do modelo
log.title("Summary of tunning")

model.summary()

# ----------------------------------------------------------------------------------
# Avaliando performance do modelo na base de treinamento

log.title("Evaluating model with training data")

data_train = load_dataframe_from_parquet(
    file_path=input_dataset_train_file_path, log=log
)

data_train = model.predict(
    data=data_train,
    y_predict_column_name="output_predict_value",
)

data_train = model.predict_class(
    data=data_train,
    trigger=0.5,
    y_predict_class_column_name="output_predict_class",
)

model.evaluate(
    data=data_train,
    y_predict_column_name="output_predict_value",
    y_target_column_name=model_io_config.target[0].name,
    view_charts=view_charts,
    save_charts=save_charts,
    results_folder=results_folder,
)


# ----------------------------------------------------------------------------------
# Avaliando performance do modelo na base de teste

log.title("Evaluating model with test data")

data_test = load_dataframe_from_parquet(file_path=input_dataset_test_file_path, log=log)
data_test["target_label"] = np.where(data_test["quality"] <= 5, "bad", "good")

data_test = model.predict(
    data=data_test,
    y_predict_column_name="output_predict",
)

model.evaluate(
    data=data_test,
    y_predict_column_name="output_predict",
    y_target_column_name=model_io_config.target[0].name,
    view_charts=view_charts,
    save_charts=save_charts,
    results_folder=results_folder,
)

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

model.save(path=output_model_file_path)

save_dataframe_to_parquet(
    data=data_train,
    file_path=output_dataset_train_predict_file_path,
    log=log,
)

save_dataframe_to_parquet(
    data=data_test,
    file_path=output_dataset_test_predict_file_path,
    log=log,
)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
