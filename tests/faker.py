import numpy as np
from src.data.make_dataset import read_data


def get_fake_data(size = 100, path = "data/raw/train.csv"):
    data_true = read_data(path)
    means = data_true.mean().values
    stds = data_true.std().values
    data_fake = []

    for mean, std in zip(means, stds):
        data_fake.append(np.random.normal(mean, std, size).reshape(-1, 1))
    return  np.hstack(data_fake)


