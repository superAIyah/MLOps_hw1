import unittest
import pandas as pd
from src.predict_pipeline import *

class TestTrainTest(unittest.TestCase):
    def test_predict(self):
        config_path = "configs/predict_config.yaml"
        data_path = "data/raw/test.csv"
        n_true = pd.read_csv(data_path).shape[0]
        pred_path = predict_pipeline(config_path)
        n_pred = pd.read_csv(pred_path).shape[0]
        self.assertEqual(n_true, n_pred)
