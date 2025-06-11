import unittest
from test_utils import prepare_data_limited
from training import train_and_save_models
from evaluation import evaluate_model
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# This file contains unit tests for the flat prices pipeline, including data preparation, model training, and evaluation.


class TestFlatPricesPipeline(unittest.TestCase):
    
    def test_data_preparation(self):
        df, X_train, X_test, y_train, y_test = prepare_data_limited()
        self.assertGreater(len(df), 0)
        self.assertEqual(X_train.shape[0] + X_test.shape[0], df.shape[0])
        self.assertFalse(X_train.isnull().values.any())
        self.assertFalse(X_test.isnull().values.any())

    def test_model_training(self):
        _, X_train, X_test, y_train, y_test = prepare_data_limited()
        models = train_and_save_models(X_train, y_train)
        self.assertIn("Linear Regression", models)
        self.assertIn("Decision Tree", models)
        self.assertIn("Random Forest", models)
        for model in models.values():
            self.assertTrue(hasattr(model, "predict"))

    def test_model_evaluation_output(self):
        _, X_train, X_test, y_train, y_test = prepare_data_limited()
        models = train_and_save_models(X_train, y_train)
        metrics = evaluate_model("Random Forest", models["Random Forest"], X_test, y_test)
        self.assertIn("MAE", metrics)
        self.assertIn("MSE", metrics)
        self.assertIn("R2", metrics)
        self.assertTrue(isinstance(metrics["R2"], float))

if __name__ == "__main__":
    unittest.main()
