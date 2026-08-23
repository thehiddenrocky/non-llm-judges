import mlflow
import mlflow.sklearn
from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import numpy as np
import os

# Ensure we are using a SQLite database for tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def run_experiment(name, ngram_range=(1, 1), class_weight=None, threshold=0.0, max_features=10000):
    with mlflow.start_run(run_name=name):
        print(f"\n--- Running Experiment: {name} ---")

        # Log params
        mlflow.log_param("ngram_range", str(ngram_range))
        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("max_features", max_features)

        # Feature Extraction
        print(f"Vectorizing with ngram_range={ngram_range}, max_features={max_features}...")
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words='english')
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Model Training
        print("Fitting Linear SVM...")
        clf = LinearSVC(random_state=42, tol=1e-5, class_weight=class_weight)
        clf.fit(X_train_tfidf, y_train)

        # Evaluation
        if threshold == 0.0:
            y_pred = clf.predict(X_test_tfidf)
        else:
            # Shift decision threshold (LinearSVC decision_function returns signed distance)
            decision_scores = clf.decision_function(X_test_tfidf)
            y_pred = (decision_scores > threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(clf, "model")

        print(f"Results for {name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return {"name": name, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    # Data Loading (Global)
    print("--- Loading Toxic-Chat Dataset ---")
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])

    X_train = df_train['user_input'].fillna("")
    y_train = df_train['toxicity']
    X_test = df_test['user_input'].fillna("")
    y_test = df_test['toxicity']

    # Set experiment name
    mlflow.set_experiment("Toxicity SVM")

    # Run 1: Base Model
    run_experiment("Base Model", 
                   ngram_range=(1, 1), 
                   class_weight=None, 
                   threshold=0.0, 
                   max_features=10000)

    # Run 2: Improved Model (Recall Focus)
    # Using 1-2 grams and balanced class weights, plus a lowered threshold
    run_experiment("Improved Model (Recall Focus)",
                   ngram_range=(1, 2),
                   class_weight='balanced',
                   threshold=-0.2,
                   max_features=20000)

    # Grid Search: Threshold + Class Weight Sweep to Optimize F1
    print("\n\n=== GRID SEARCH: THRESHOLD & CLASS WEIGHT ===")
    thresholds = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2]
    class_weights = [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]

    results = []
    for cw in class_weights:
        for thresh in thresholds:
            if cw is None:
                cw_label = "None"
            elif isinstance(cw, str):
                cw_label = cw
            else:
                cw_label = f"custom_{cw[1]}"
            name = f"GridSearch - CW:{cw_label} Threshold:{thresh}"
            result = run_experiment(name,
                                   ngram_range=(1, 2),
                                   class_weight=cw,
                                   threshold=thresh,
                                   max_features=20000)
            results.append(result)

    # Find and display best F1
    print("\n\n=== GRID SEARCH RESULTS ===")
    best_result = max(results, key=lambda x: x["f1"])
    print(f"\nBest F1 Score: {best_result['f1']:.4f}")
    print(f"Best Model: {best_result['name']}")
    print(f"  Accuracy:  {best_result['accuracy']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
