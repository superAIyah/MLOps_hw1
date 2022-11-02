from src.features.CustomTransformer import CustomTransformer
from .faker import get_fake_data
import unittest


class TestsLruFileGen(unittest.TestCase):

    def test_shape(self):
        data = get_fake_data()
        scalers1 = ["MinMaxScaler", "PolynomialFeatures"]
        scalers2 = ["PolynomialFeatures", "StandardScaler"]
        trans1 = CustomTransformer(scalers1)
        trans2 = CustomTransformer(scalers2)
        result1 = trans1.fit_transform(data)
        result2 = trans2.fit_transform(data)
        self.assertEqual(result1.shape, result2.shape)

