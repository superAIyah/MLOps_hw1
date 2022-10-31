import pandas as pd
from src.entities.feature_params import FeatureParams
from .CustomTransformer import CustomTransformer


def build_transformer(params: FeatureParams) -> CustomTransformer:
    return CustomTransformer(params.transformers)


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target
