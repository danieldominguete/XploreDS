"""
Xplore DS :: Training Logistic Regression Model for Scoring Classification
"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib em projeto local
# Futuramente substituir pois a lib estará já instalada no .venv
project_folder = Path(__file__).resolve().parents[6]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xploreds.environment.environment import XploreDSLocalhost
from xploreds.environment.logging import XploreDSLogging
from xploreds.data_handler.file import (
    load_dataframe_from_parquet,
    save_dataframe_to_parquet,
)
from xploreds.models.logistic_regression import XLogisticRegression
from xploreds.data_schemas.logistic_regression_config import (
    LogisticRegressionArchiteture,
    LogisticRegressionHyperparameters,
    Topology,
    FitAlgorithm,
)
from xploreds.data_schemas.model_io_config import (
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
# Configuracao de master table de entrada

input_dataset_train_file_path = "data/credit-g/processed/credit-g_train.parquet"
input_dataset_test_file_path = "data/credit-g/processed/credit-g_test.parquet"

# ----------------------------------------------------------------------------------
# Configuracao de parametros de I/O do modelo

features = [
    VariableConfig(name="duration", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="credit_amount", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(
        name="installment_commitment", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(name="residence_since", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="age", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="num_dependents", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(
        name="checking_status_0<=X<200", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(name="checking_status_<0", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(
        name="checking_status_>=200", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="checking_status_no checking", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="credit_history_all paid", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="credit_history_critical/other existing credit",
        scaling_method=ScalingMethod.none_scaler,
    ),
    VariableConfig(
        name="credit_history_delayed previously",
        scaling_method=ScalingMethod.none_scaler,
    ),
    VariableConfig(
        name="credit_history_existing paid", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="credit_history_no credits/all paid",
        scaling_method=ScalingMethod.none_scaler,
    ),
    VariableConfig(name="purpose_business", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(
        name="purpose_domestic appliance", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(name="purpose_education", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(
        name="purpose_furniture/equipment", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(name="purpose_new car", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="purpose_other", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="purpose_radio/tv", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="purpose_repairs", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="purpose_retraining", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="purpose_used car", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(
        name="savings_status_100<=X<500", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="savings_status_500<=X<1000", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="savings_status_<100", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="savings_status_>=1000", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="savings_status_no known savings", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(name="employment_1<=X<4", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="employment_4<=X<7", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="employment_<1", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="employment_>=7", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(
        name="employment_unemployed", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="personal_status_female div/dep/mar",
        scaling_method=ScalingMethod.none_scaler,
    ),
    VariableConfig(
        name="personal_status_male div/sep", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="personal_status_male mar/wid", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="personal_status_male single", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="other_parties_co applicant", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="other_parties_guarantor", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(name="other_parties_none", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(
        name="property_magnitude_car", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="property_magnitude_life insurance",
        scaling_method=ScalingMethod.none_scaler,
    ),
    VariableConfig(
        name="property_magnitude_no known property",
        scaling_method=ScalingMethod.none_scaler,
    ),
    VariableConfig(
        name="property_magnitude_real estate", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="other_payment_plans_bank", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="other_payment_plans_none", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="other_payment_plans_stores", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(name="housing_for free", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="housing_own", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="housing_rent", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(
        name="job_high qualif/self emp/mgmt", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(name="job_skilled", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(
        name="job_unemp/unskilled non res", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(
        name="job_unskilled resident", scaling_method=ScalingMethod.none_scaler
    ),
    VariableConfig(name="own_telephone_none", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="own_telephone_yes", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="foreign_worker_no", scaling_method=ScalingMethod.none_scaler),
    VariableConfig(name="foreign_worker_yes", scaling_method=ScalingMethod.none_scaler),
]

model_io_config = ModelIOConfig(
    application_type=ApplicationType.scoring_classification,
    date_reference="transaction_date_month",
    features=features,
    target_numerical=[VariableConfig(name="class_bad")],
    target_categorical_index=VariableConfig(name="class_bad"),
    target_categorical_label=VariableConfig(name="class"),
    target_categorical_index_to_label={1: "bad", 0: "good"},
)

# ----------------------------------------------------------------------------------
# Setup do modelo

model_config = LogisticRegressionArchiteture(
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
    results_folder + "data/credit-g/processed/credit-g_train_predict.parquet"
)
output_dataset_test_predict_file_path = (
    results_folder + "data/credit-g/processed/credit-g_test_predict.parquet"
)
output_model_file_path = results_folder + "models/credit-g_logistic_regression.joblib"

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

log.title("Tunning model")

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

log.info("Predicting output value ...")
data_train = model.predict(
    data=data_train,
    y_predict_column_name_output="output_predict_value",
)

model.evaluate(
    data=data_train,
    y_predict_numerical_column_list=["output_predict_value"],
    y_target_numerical_column_list=[model_io_config.target_numerical[0].name],
    date_reference_column_name=model_io_config.date_reference,
    dataset_identification="train",
    view_charts=view_charts,
    save_charts=save_charts,
    results_folder=results_folder,
)


# ----------------------------------------------------------------------------------
# Avaliando performance do modelo na base de teste

log.title("Evaluating model with test data")

data_test = load_dataframe_from_parquet(file_path=input_dataset_test_file_path, log=log)


data_test = model.predict(
    data=data_test,
    y_predict_column_name_output="output_predict_value",
)


model.evaluate(
    data=data_test,
    y_predict_numerical_column_list=["output_predict_value"],
    y_target_numerical_column_list=[model_io_config.target_numerical[0].name],
    date_reference_column_name=model_io_config.date_reference,
    dataset_identification="test",
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
