# -*- coding: utf-8 -*-
import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.entities import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(f"{dir_path=}")
    data = pd.read_csv(path)
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data
