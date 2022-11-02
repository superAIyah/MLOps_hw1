from typing import Optional

from dataclasses import dataclass

from .download_params import DownloadParams
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import TrainingParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    train_params: TrainingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    downloading_params: Optional[DownloadParams] = None
    use_mlflow: bool = True
    mlflow_experiment: str = "Model inference"


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        print("Opened")
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
