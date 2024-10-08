"""
Xplore DS :: Configuration data structure 
"""

from pydantic import BaseModel
from enum import Enum


class FitAlgorithm(str, Enum):

    ordinary_least_squares = "ordinary_least_squares"
    generalized_least_squares = "generalized_least_squares"


class LinearRegressionConfig(BaseModel):

    set_intersection_with_zero: bool = False


class LinearRegressionHyperparameters(BaseModel):

    fit_algorithm: FitAlgorithm = FitAlgorithm.ordinary_least_squares
