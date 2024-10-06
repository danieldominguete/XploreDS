"""
Xplore DS :: Training Linear Regression Model
"""

# Importando bibliotecas nativas
import sys, os
from pathlib import Path
from dotenv import load_dotenv


# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[8]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xplore_ds.environment.environment import XploreDSLocalhost
from xplore_ds.environment.logging import XploreDSLogging
from xplore_ds.data_handler.file import load_dataframe_from_parquet
from xplore_ds.models.linear_regression import XLinearRegression
import xplore_ds.data_schemas.linear_regression_config as config
from xplore_ds.data_schemas.knowledge_base_config import (
    KnowledgeBaseSetup,
    FeaturesSetup,
    TargetSetup,
    ScalingMethod,
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

# Seeds
random_state = 100

# ==================================================================================
# Parametrizacao do script
# ==================================================================================

log.title("Script setup")

# Configuracao de dados de entrada
input_dataset_train_file_path = (
    "data/projects/stage/wine_quality/wine_quality_train.parquet"
)
input_dataset_test_file_path = (
    "data/projects/stage/wine_quality/wine_quality_test.parquet"
)

# Setup da base de conhecimento

# configuracao de cada feature
fixed_acidity = FeaturesSetup(name="fixed acidity", scaler=ScalingMethod.none_scaler)
volatile_acidity = FeaturesSetup(
    name="volatile acidity", scaler=ScalingMethod.min_max_scaler
)
citric_acid = FeaturesSetup(name="citric acid", scaler=ScalingMethod.mean_std_scaler)
residual_sugar = FeaturesSetup(name="residual sugar", scaler=None)
chlorides = FeaturesSetup(name="chlorides", scaler=None)
free_sulfur_dioxide = FeaturesSetup(name="free sulfur dioxide", scaler=None)
total_sulfur_dioxide = FeaturesSetup(name="total sulfur dioxide", scaler=None)
density = FeaturesSetup(name="density", scaler=None)
pH = FeaturesSetup(name="pH", scaler=None)
sulphates = FeaturesSetup(name="sulphates", scaler=None)
alcohol = FeaturesSetup(name="alcohol", scaler=None)
quality = TargetSetup(name="quality")

knowledge_base_setup = KnowledgeBaseSetup(
    features=[fixed_acidity, volatile_acidity, citric_acid, residual_sugar],
    target=quality,
)

# Setup do modelo
model_setup = config.LinearRegressionSetup(set_intersection_with_zero=False)

# Hiperparametros
model_hyperparameters = config.LinearRegressionHyperparameters(
    fit_algorithm=config.FitAlgorithm.ordinary_least_squares,
)

# Configuracao de dados de saida
output_folder = "output"
output_dataset_predict_file_path = (
    "data/projects/prod/wine_quality/wine_quality_train_predict.parquet"
)
output_model_file_path = (
    "static/models/wine_quality/wine_quality_linear_regression.joblib"
)

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

model = XLinearRegression(
    setup=model_setup, hyperparameters=model_hyperparameters, log=log
)
model.train(data=data_train, features_column_name=features, target_column=target_column)
model.summary()

log.title("Evaluating model")
data_test = load_dataframe_from_parquet(file_path=input_dataset_test_file_path, log=log)

data_test = model.predict(
    data_test=data_test,
    features_column_name=features,
    y_predict_column_name="output_predict",
)

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")
model.save(path=output_model_file_path)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
