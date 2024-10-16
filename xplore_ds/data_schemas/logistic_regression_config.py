"""
Xplore DS :: Configuration data structure
"""

from pydantic import BaseModel
from enum import Enum


class Topology(str, Enum):

    logit = "logit"
    probit = "probit"


class FitAlgorithm(str, Enum):

    maximum_likelihood = "maximum_likelihood"


class LogisticRegressionArchiteture(BaseModel):

    set_intersection_with_zero: bool = False
    topology: Topology = Topology.logit


class LogisticRegressionHyperparameters(BaseModel):

    fit_algorithm: FitAlgorithm = FitAlgorithm.maximum_likelihood
