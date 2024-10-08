"""
Xplore DS :: Configuration data structure 
"""

from pydantic import BaseModel
from enum import Enum


class ScalingMethod(str, Enum):

    none_scaler = "none_scaler"
    min_max_scaler = "min_max_scaler"
    mean_std_scaler = "mean_std_scaler"


class FeaturesConfig(BaseModel):
    name: str
    scaling_method: ScalingMethod = ScalingMethod.none_scaler


class TargetConfig(BaseModel):
    name: str
    scaling_method: ScalingMethod = ScalingMethod.none_scaler


class KnowledgeBaseConfig(BaseModel):

    features: list[FeaturesConfig]
    target: list[TargetConfig]
