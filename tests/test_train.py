import unittest
import pandas as pd
from src.train_pipeline import *

class TestTrainTest(unittest.TestCase):
    def test_read(self):
        data = read_data("data/raw/train.csv")
        self.assertIsInstance(data, pd.DataFrame)

    def test_pipeline(self):
        config_path = "configs/train_config.yaml"
        training_pipeline_params = read_training_pipeline_params(config_path)
        data = read_data(training_pipeline_params.input_data_path)
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
        transformer = build_transformer(training_pipeline_params.feature_params)
        transformer.fit(train_df)
        train_features = transformer.transform(train_df)
        model = train_model(
            train_features, train_target, training_pipeline_params.train_params
        )
        inference_pipeline = create_inference_pipeline(model, transformer)
        predicts, probs = predict_model(inference_pipeline, val_df)
        metrics = evaluate_model(predicts, probs, val_target)
        self.assertEqual(predicts.shape, val_target.shape)
        metrics_val = metrics.values()
        self.assertTrue(all([val > 0 for val in metrics_val]))