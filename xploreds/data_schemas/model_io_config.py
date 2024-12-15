"""
Xplore DS :: Configuration data structure
"""

from pydantic import BaseModel
from enum import Enum
from xploreds.data_schemas.pre_processing_config import ScalingMethod, EncodingMethod


class ApplicationType(str, Enum):

    clustering = "clustering"
    regression = "regression"
    scoring_classification = "scoring_classification"
    binary_classification = "binary_classification"
    multiclass_classification = "multiclass_classification"


class VariableConfig(BaseModel):
    name: str
    scaling_method: ScalingMethod = ScalingMethod.none_scaler
    encoding_method: EncodingMethod = EncodingMethod.none_encoder


class ModelIOConfig(BaseModel):

    application_type: ApplicationType
    features: list[VariableConfig]
    target_numerical: list[VariableConfig]
    target_categorical_label: VariableConfig = None
    target_categorical_index: VariableConfig = None
    target_categorical_index_to_label: dict[int, str] = {}
    date_reference: str = None
