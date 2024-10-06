"""
Xplore DS :: Configuration data structure 
"""

from pydantic import BaseModel
from enum import Enum


class ScalingMethod(str, Enum):

    none_scaler = "none_scaler"
    min_max_scaler = "min_max_scaler"
    mean_std_scaler = "mean_std_scaler"


class FeaturesSetup(BaseModel):
    name: str
    scaling_method: ScalingMethod = ScalingMethod.none_scaler


class TargetSetup(BaseModel):
    name: str


class KnowledgeBaseSetup(BaseModel):

    features: list[FeaturesSetup]
    target: TargetSetup
