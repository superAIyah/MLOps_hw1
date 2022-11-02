import numpy as np
import pandas as pd
import pickle
from typing import Dict, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from src.entities.train_params import TrainingParams
from src.features import CustomTransformer

SklearnClassifierModel = Union[
    RandomForestClassifier, LogisticRegression, KNeighborsClassifier
]


def predict_model(
    model: Pipeline, features: pd.DataFrame
) -> (np.ndarray, np.ndarray):
    predicts = model.predict(features)
    predicts_prob = model.predict_proba(features)
    return predicts, predicts_prob


def evaluate_model(
    predicts: np.ndarray, predicts_prob: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    return {
        "accuracy score": accuracy_score(target, predicts),
        "f1 score": f1_score(target, predicts),
        "ROC-AUC score": roc_auc_score(target, predicts_prob[:, 1])
    }


def create_inference_pipeline(
    model: SklearnClassifierModel, transformer: CustomTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassifierModel:
    model_dict = {
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42),
        "KNN": KNeighborsClassifier()
    }
    model = model_dict[train_params.model_type]
    model.fit(features, target)
    return model


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(
    model_path: str
) -> Pipeline:
    model = pickle.load(open(model_path, 'rb'))
    return model
