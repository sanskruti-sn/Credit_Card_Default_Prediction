import unittest
import os
import pickle
import numpy as np
import pandas as pd

from src import data_ingestion, preprocessing, model


class TestModelLoading(unittest.TestCase):
    def setUp(self):
        model_path = os.path.join('notebooks', 'artifacts', 'model.pkl')
        scaler_path = os.path.join('notebooks', 'artifacts', 'scaler.pkl')

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def test_model_load(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.scaler)

    def test_prediction(self):
        dummy_features = np.random.rand(1, 23)  # Adjust feature count as per your model
        scaled_features = self.scaler.transform(dummy_features)
        prediction = self.model.predict(scaled_features)

        self.assertEqual(prediction.shape, (1,))
        self.assertIn(prediction[0], [0, 1])


class TestDataIngestion(unittest.TestCase):
    def test_load_data(self):
        df = data_ingestion.load_data()
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        self.assertIn('default.payment.next.month', df.columns)


class TestPreprocessing(unittest.TestCase):
    def test_preprocess(self):
        df = pd.DataFrame({
            'feature1': [1, 2, None, 4],
            'feature2': [1, 0, 1, 0],
            'default.payment.next.month': [0, 1, 0, 1]
        })

        result = preprocessing.preprocess_data(df)
        if isinstance(result, tuple):
            processed_data = result[0]
        else:
            processed_data = result

        # Check if processed_data is numpy array or DataFrame
        if isinstance(processed_data, np.ndarray):
            # For numpy array, check for NaN using np.isnan
            self.assertFalse(np.isnan(processed_data).any())
        else:
            # For DataFrame, use isnull
            self.assertFalse(processed_data.isnull().any().any())


class TestModelFunctions(unittest.TestCase):
    def test_train_model(self):
        X_train = pd.DataFrame(np.random.rand(10, 5))
        y_train = pd.Series(np.random.randint(0, 2, size=10))
        X_test = pd.DataFrame(np.random.rand(5, 5))
        y_test = pd.Series(np.random.randint(0, 2, size=5))

        clf = model.train_model(X_train, y_train, X_test, y_test)
        self.assertIsNotNone(clf)
        self.assertTrue(hasattr(clf, 'predict'))

    # Commented out because evaluate_model does not exist in your project code
    # def test_evaluate_model(self):
    #     X_train = pd.DataFrame(np.random.rand(10, 5))
    #     y_train = pd.Series(np.random.randint(0, 2, size=10))
    #     X_test = pd.DataFrame(np.random.rand(5, 5))
    #     y_test = pd.Series(np.random.randint(0, 2, size=5))

    #     clf = model.train_model(X_train, y_train, X_test, y_test)
    #     metrics = model.evaluate_model(clf, X_test, y_test)
    #     self.assertIn('accuracy', metrics)
    #     self.assertGreaterEqual(metrics['accuracy'], 0)
    #     self.assertLessEqual(metrics['accuracy'], 1)


if __name__ == '__main__':
    unittest.main()
