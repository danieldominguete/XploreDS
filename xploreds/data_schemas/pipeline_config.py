"""
Xplore DS :: Pipeline configuration
"""

from pydantic import BaseModel
from enum import Enum
from typing import Optional
from xplore_ds.data_schemas.model_io_config import ModelIOConfig


class PipelineType(str, Enum):

    model_tunning = "model_tunning"


class ModelType(str, Enum):

    linear_regression = "linear_regression"
    logistic_regression = "logistic_regression"


class PipelineConfig(BaseModel):

    pipeline_label: str
    pipeline_description: Optional[str] = None
    pipeline_type: PipelineType
    view_charts: bool = False
    save_charts: bool = True


class PipelineModelTunningConfig(BaseModel):

    input_dataset_train_file_path: str
    input_dataset_test_file_path: str
    model_io_config: ModelIOConfig
    model_type: ModelType
    model_architeture: dict
    model_hyperparameters: dict
