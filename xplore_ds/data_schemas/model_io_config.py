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


class VariableType(str, Enum):

    numerical = "numerical"
    categorical = "categorical"
    textual = "textual"


class ScalingMethod(str, Enum):

    none_scaler = "none_scaler"
    min_max_scaler = "min_max_scaler"
    mean_std_scaler = "mean_std_scaler"


class VariableConfig(BaseModel):
    name: str
    scaling_method: ScalingMethod = ScalingMethod.none_scaler


class ModelIOConfig(BaseModel):

    application_type: ApplicationType
    features: list[VariableConfig]
    target: list[VariableConfig]
