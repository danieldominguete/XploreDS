"""
Xplore DS :: Configuration data structure
"""

from pydantic import BaseModel
from enum import Enum


class VariableType(str, Enum):

    numerical = "numerical"
    categorical = "categorical"
    textual = "textual"


class EncodingMethod(str, Enum):

    none_encoder = "none_encoder"
    one_hot_encoder = "one_hot_encoder"


class VariableConfig(BaseModel):
    name: str
    encoding_method: EncodingMethod = EncodingMethod.none_encoder


class DatasetConfig(BaseModel):

    variables: list[VariableConfig]
