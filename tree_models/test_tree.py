import os
import unittest
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from tree_models.train_tree import train_and_eval

class TestTreeModels(unittest.TestCase):
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
        self.y_train = np.array([0, 0, 1, 1, 0, 0])
        
        self.X_test = [
            "This is another safe prompt.",
            "You are incredibly stupid"  # mock toxic
        ]
        self.y_test = np.array([0, 1])

    def test_train_and_eval_xgboost(self):
        # Test that train_and_eval runs with xgboost on mock data
        result = train_and_eval(
            model_type="xgboost",
            run_name="Test-XGBoost",
            ngram_range=(1, 1),
            max_features=100,
            n_estimators=5,
            max_depth=2,
            learning_rate=0.1,
            scale_pos_weight=1.0,
            threshold=0.5,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test
        )
        
        self.assertEqual(result["model_type"], "xgboost")
        self.assertIn("accuracy", result)
        self.assertIn("f1", result)

    def test_train_and_eval_lightgbm(self):
        # Test that train_and_eval runs with lightgbm on mock data
        result = train_and_eval(
            model_type="lightgbm",
            run_name="Test-LightGBM",
            ngram_range=(1, 1),
            max_features=100,
            n_estimators=5,
            max_depth=2,
            learning_rate=0.1,
            scale_pos_weight=1.0,
            threshold=0.5,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test
        )
        
        self.assertEqual(result["model_type"], "lightgbm")
        self.assertIn("accuracy", result)
        self.assertIn("f1", result)

    def test_threshold_shifting(self):
        # Test custom probability threshold shifting manually
        vectorizer = TfidfVectorizer(max_features=50)
        X_train_tfidf = vectorizer.fit_transform(self.X_train)
        X_test_tfidf = vectorizer.transform(self.X_test)
        
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=5, random_state=42, verbose=-1)
        clf.fit(X_train_tfidf, self.y_train)
        
        probs = clf.predict_proba(X_test_tfidf)[:, 1]
        
        # Shift threshold to 0.1
        preds_low = (probs >= 0.1).astype(int)
        # Shift threshold to 0.9
        preds_high = (probs >= 0.9).astype(int)
        
        self.assertEqual(len(preds_low), len(self.X_test))
        self.assertTrue(np.all((preds_low >= preds_high)))

if __name__ == "__main__":
    unittest.main()
