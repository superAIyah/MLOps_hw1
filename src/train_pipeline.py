import logging
import sys
import warnings
import json
import click

from src.data import read_data, split_train_val_data
from src.entities.train_pipeline_params import read_training_pipeline_params
from src.features.build_features import extract_target, build_transformer
from src.models import (
    train_model,
    predict_model,
    evaluate_model,
    serialize_model
)
from src.models.model_fit_predict import create_inference_pipeline

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)
    return run_train_pipeline(training_pipeline_params)


def run_train_pipeline(training_pipeline_params):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
            data, training_pipeline_params.splitting_params
        )

    val_target = extract_target(
        val_df, training_pipeline_params.feature_params
    )
    train_target = extract_target(
        train_df, training_pipeline_params.feature_params
    )
    train_df = train_df.drop(
        columns=training_pipeline_params.feature_params.target_col
    )
    val_df = val_df.drop(
        columns=training_pipeline_params.feature_params.target_col
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    train_features = transformer.transform(train_df)
    logger.info(f"train_features.shape is {train_features.shape}")
    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )
    inference_pipeline = create_inference_pipeline(model, transformer)
    predicts, probs = predict_model(inference_pipeline, val_df)
    metrics = evaluate_model(predicts, probs, val_target)
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")
    path_to_model = serialize_model(
        inference_pipeline, training_pipeline_params.output_model_path
    )
    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
