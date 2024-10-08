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
    KnowledgeBaseConfig,
    FeaturesConfig,
    TargetConfig,
    ScalingMethod,
    ApplicationType,
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

# configuracao de features e target

volatile_acidity = FeaturesConfig(
    name="volatile acidity", scaling_method=ScalingMethod.min_max_scaler
)
citric_acid = FeaturesConfig(
    name="citric acid", scaling_method=ScalingMethod.mean_std_scaler
)
residual_sugar = FeaturesConfig(
    name="residual sugar", scaling_method=ScalingMethod.none_scaler
)
chlorides = FeaturesConfig(name="chlorides", scaling_method=ScalingMethod.none_scaler)
free_sulfur_dioxide = FeaturesConfig(
    name="free sulfur dioxide", scaling_method=ScalingMethod.none_scaler
)
total_sulfur_dioxide = FeaturesConfig(
    name="total sulfur dioxide", scaling_method=ScalingMethod.none_scaler
)
density = FeaturesConfig(name="density", scaling_method=ScalingMethod.none_scaler)
pH = FeaturesConfig(name="pH", scaling_method=ScalingMethod.none_scaler)
sulphates = FeaturesConfig(name="sulphates", scaling_method=ScalingMethod.none_scaler)
alcohol = FeaturesConfig(name="alcohol", scaling_method=ScalingMethod.none_scaler)
fixed_acidity = FeaturesConfig(
    name="fixed acidity", scaling_method=ScalingMethod.none_scaler
)
quality = TargetConfig(name="quality")

knowledge_base_config = KnowledgeBaseConfig(
    application_type=ApplicationType.regression,
    features=[
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol,
        fixed_acidity,
    ],
    target=[quality],
)

# Setup do modelo
model_config = config.LinearRegressionConfig(set_intersection_with_zero=False)

# Hiperparametros
tunning_config = config.LinearRegressionHyperparameters(
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
view_charts = True
save_charts = True
results_folder = log.log_path

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

# criando objeto modelo
model = XLinearRegression(
    kb_config=knowledge_base_config,
    model_config=model_config,
    tunning_config=tunning_config,
    random_state=random_state,
    log=log,
)


log.title("Training model")

model.fit(data=data_train)
model.summary()

log.title("Evaluating model with test data")

data_test = load_dataframe_from_parquet(file_path=input_dataset_test_file_path, log=log)

data_test = model.predict(
    data=data_test,
    y_predict_column_name="output_predict",
)

model.evaluate(
    data=data_test,
    y_predict_column_name="output_predict",
    y_target_column_name=knowledge_base_config.target[0].name,
    view_charts=view_charts,
    save_charts=save_charts,
    results_folder=results_folder,
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
