import os
import unittest
import tempfile
import numpy as np
from fasttext_models.train_fasttext import train_and_eval, prepare_fasttext_file, evaluate_predictions

class TestFastTextModel(unittest.TestCase):
    def setUp(self):
        # Create a small mock dataset
        self.X_train = [
            "This is a perfectly safe chat message.",
            "I love programming in Python.",
            "Kill yourself and go away",  # mock toxic
            "You are an idiot and stupid",  # mock toxic
            "Have a nice day!",
            "Let's play a video game."
        ]
        self.y_train = [0, 0, 1, 1, 0, 0]
        
        self.X_test = [
            "This is another safe prompt.",
            "You are incredibly stupid"  # mock toxic
        ]
        self.y_test = [0, 1]

        # Prepare a temporary file for fasttext
        self.temp_dir = tempfile.gettempdir()
        self.train_file = os.path.join(self.temp_dir, "fasttext_test_mock_train.txt")
        prepare_fasttext_file(self.X_train, self.y_train, self.train_file)

    def tearDown(self):
        if os.path.exists(self.train_file):
            os.remove(self.train_file)

    def test_prepare_fasttext_file(self):
        # Test file exists and contains expected formatting
        self.assertTrue(os.path.exists(self.train_file))
        with open(self.train_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), len(self.X_train))
        self.assertTrue(lines[0].startswith("__label__0"))
        self.assertTrue(lines[2].startswith("__label__1"))

    def test_train_and_eval(self):
        # Run training and evaluation with MLflow tracking disabled or tracked with a custom run name
        result = train_and_eval(
            run_name="Test-FastText",
            learning_rate=0.1,
            epochs=2,
            wordNgrams=1,
            dim=10,
            threshold=0.5,
            train_file=self.train_file,
            X_test=self.X_test,
            y_test=self.y_test,
            log_model=False  # Do not log test artifacts to MLflow
        )

        self.assertIn("accuracy", result)
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1", result)
        self.assertEqual(result["learning_rate"], 0.1)
        self.assertEqual(result["epochs"], 2)

    def test_threshold_shifting(self):
        # Mock probabilities
        y_prob = np.array([0.2, 0.8])
        y_test = np.array([0, 1])

        # Test evaluation at threshold 0.5
        acc, prec, rec, f1, y_pred = evaluate_predictions(y_test, y_prob, 0.5)
        self.assertEqual(acc, 1.0)
        self.assertTrue(np.array_equal(y_pred, [0, 1]))

        # Test evaluation at threshold 0.9 (both should be class 0)
        acc, prec, rec, f1, y_pred = evaluate_predictions(y_test, y_prob, 0.9)
        self.assertEqual(acc, 0.5)
        self.assertTrue(np.array_equal(y_pred, [0, 0]))

if __name__ == "__main__":
    unittest.main()
