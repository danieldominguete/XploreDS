"""
Xplore DS :: Configuration data structure
"""

from pydantic import BaseModel
from enum import Enum


class ApplicationType(str, Enum):

    regression = "regression"
    binary_classification = "binary_classification"
    multiclass_classification = "multiclass_classification"
    clustering = "clustering"


class ScalingMethod(str, Enum):

    none_scaler = "none_scaler"
    min_max_scaler = "min_max_scaler"
    mean_std_scaler = "mean_std_scaler"


class VariableIOConfig(BaseModel):
    name: str
    scaling_method: ScalingMethod = ScalingMethod.none_scaler


class ModelIOConfig(BaseModel):

    application_type: ApplicationType
    features: list[VariableIOConfig]
    target_numerical: list[VariableIOConfig]
    target_categorical_index: list[VariableIOConfig] = []
    target_textual: list[VariableIOConfig] = []
    target_index_to_label: dict[int, str] = {}
