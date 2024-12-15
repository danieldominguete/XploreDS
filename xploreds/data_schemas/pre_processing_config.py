"""
Xplore DS :: Configuration data structure
"""

from pydantic import BaseModel
from enum import Enum


class ScalingMethod(str, Enum):

    none_scaler = "none_scaler"
    min_max_scaler = "min_max_scaler"
    mean_std_scaler = "mean_std_scaler"


class EncodingMethod(str, Enum):

    none_encoder = "none_encoding"
    one_hot_encoder = "one_hot_encoder"
