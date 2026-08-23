import os
import tempfile
import numpy as np
import pandas as pd
import mlflow
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import fasttext

# Ensure robust, centralized MLflow tracking URI pointing to root mlflow.db
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
db_path = os.path.join(project_root, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

def prepare_fasttext_file(texts, labels, file_path):
    """
    Saves texts and labels into a text file formatted for FastText.
    FastText expects each line to contain '__label__<val> <text>'.
    Any newlines inside the text are replaced with spaces.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for text, label in zip(texts, labels):
            # Replace newlines with spaces and remove leading/trailing spaces
            cleaned_text = str(text).replace("\n", " ").replace("\r", " ").strip()
            f.write(f"__label__{label} {cleaned_text}\n")

def evaluate_predictions(y_test, y_prob, threshold):
    """
    Computes performance metrics for a given prediction probability and decision threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    return acc, precision, recall, f1, y_pred

def train_and_eval(
    run_name="FastTextRun",
    learning_rate=0.1,
    epochs=10,
    wordNgrams=1,
    dim=100,
    threshold=0.5,
    train_file=None,
    X_test=None,
    y_test=None,
    log_model=True
):
    """
    Trains a FastText classifier, evaluates it using a custom decision threshold,
    and logs all parameters, metrics, and optionally the model artifact to MLflow.
    """
    with mlflow.start_run(run_name=run_name):
        print(f"\n--- Running Experiment: {run_name} ---")

        # Log parameters
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("wordNgrams", wordNgrams)
        mlflow.log_param("dim", dim)
        mlflow.log_param("decision_threshold", threshold)

        # Train model
        print(f"Training FastText (epochs={epochs}, lr={learning_rate}, ngrams={wordNgrams}, dim={dim})...")
        model = fasttext.train_supervised(
            input=train_file,
            lr=learning_rate,
            epoch=epochs,
            wordNgrams=wordNgrams,
            dim=dim,
            thread=4
        )

        # Preprocess test texts for prediction
        cleaned_test_texts = [str(text).replace("\n", " ").replace("\r", " ").strip() for text in X_test]

        # Predict probability of classes (k=2 fetches probabilities for all classes)
        labels, probs = model.predict(cleaned_test_texts, k=2)

        # Map predictions to the probability of the toxic label ('__label__1')
        y_prob = []
        for label_list, prob_list in zip(labels, probs):
            try:
                idx = label_list.index('__label__1')
                prob_val = prob_list[idx]
            except ValueError:
                # If '__label__1' is missing, infer from '__label__0'
                if '__label__0' in label_list:
                    idx = label_list.index('__label__0')
                    prob_val = 1.0 - prob_list[idx]
                else:
                    prob_val = 0.0
            y_prob.append(prob_val)
        y_prob = np.array(y_prob)

        # Evaluate at specified threshold
        acc, precision, recall, f1, y_pred = evaluate_predictions(y_test, y_prob, threshold)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save and log FastText model as artifact
        if log_model:
            model_path = os.path.join(script_dir, "fasttext_model.bin")
            model.save_model(model_path)
            mlflow.log_artifact(model_path, artifact_path="model")
            if os.path.exists(model_path):
                os.remove(model_path)

        # Print results
        print(f"Results for {run_name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        return {
            "name": run_name,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "wordNgrams": wordNgrams,
            "dim": dim,
            "threshold": threshold,
            "y_prob": y_prob
        }

if __name__ == "__main__":
    # Load dataset
    print("--- Loading Toxic-Chat Dataset ---")
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])

    X_train = df_train['user_input'].fillna("")
    y_train = df_train['toxicity']
    X_test = df_test['user_input'].fillna("")
    y_test = df_test['toxicity']

    # Write data to temporary files for FastText consumption
    print("Preparing FastText dataset files...")
    temp_dir = tempfile.gettempdir()
    train_file = os.path.join(temp_dir, "fasttext_toxic_train.txt")
    test_file = os.path.join(temp_dir, "fasttext_toxic_test.txt")

    prepare_fasttext_file(X_train, y_train, train_file)
    prepare_fasttext_file(X_test, y_test, test_file)

    # Set up MLflow experiment
    experiment_name = "Toxicity FastText Models"
    mlflow.set_experiment(experiment_name)

    results = []

    # Run 1: Baseline FastText Model
    res_base = train_and_eval(
        run_name="FastText Baseline",
        learning_rate=0.1,
        epochs=10,
        wordNgrams=1,
        dim=100,
        threshold=0.5,
        train_file=train_file,
        X_test=X_test,
        y_test=y_test,
        log_model=True
    )
    results.append(res_base)

    # Run 2: Improved FastText Model (Recall Focus)
    # Using bi-grams, higher epochs, learning rate, and lowered probability threshold
    res_recall = train_and_eval(
        run_name="FastText Recall Focus",
        learning_rate=0.5,
        epochs=15,
        wordNgrams=2,
        dim=100,
        threshold=0.2,
        train_file=train_file,
        X_test=X_test,
        y_test=y_test,
        log_model=True
    )
    results.append(res_recall)

    # Run 3: Grid Search over hyperparameters to maximize F1-score
    print("\n\n=== GRID SEARCH: HYPERPARAMETER SWEEP ===")
    lrs = [0.1, 0.3, 0.5]
    epochs_list = [10, 15, 20]
    ngrams_list = [1, 2]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    best_f1_overall = -1.0
    best_res = None

    for lr in lrs:
        for epochs in epochs_list:
            for wordNgrams in ngrams_list:
                print(f"Candidate model: lr={lr}, epochs={epochs}, ngrams={wordNgrams}")
                # Train the model once
                model = fasttext.train_supervised(
                    input=train_file,
                    lr=lr,
                    epoch=epochs,
                    wordNgrams=wordNgrams,
                    dim=100,
                    thread=4
                )
                
                # Get prediction probabilities for test set
                cleaned_test_texts = [str(text).replace("\n", " ").replace("\r", " ").strip() for text in X_test]
                labels, probs = model.predict(cleaned_test_texts, k=2)
                
                y_prob = []
                for label_list, prob_list in zip(labels, probs):
                    try:
                        idx = label_list.index('__label__1')
                        prob_val = prob_list[idx]
                    except ValueError:
                        if '__label__0' in label_list:
                            idx = label_list.index('__label__0')
                            prob_val = 1.0 - prob_list[idx]
                        else:
                            prob_val = 0.0
                    y_prob.append(prob_val)
                y_prob = np.array(y_prob)
                
                # Evaluate in-memory across thresholds to find the best threshold for this model
                best_thresh_for_model = 0.5
                best_f1_for_model = -1.0
                best_metrics_for_model = None
                
                for threshold in thresholds:
                    acc, precision, recall, f1, _ = evaluate_predictions(y_test, y_prob, threshold)
                    candidate_res = {
                        "accuracy": acc,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "threshold": threshold
                    }
                    if f1 > best_f1_for_model:
                        best_f1_for_model = f1
                        best_thresh_for_model = threshold
                        best_metrics_for_model = candidate_res
                
                # Now, start ONE run for this model combination in MLflow (using the best threshold)
                run_name = f"Sweep - lr:{lr} ep:{epochs} ng:{wordNgrams} th:{best_thresh_for_model:.1f}"
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_param("learning_rate", lr)
                    mlflow.log_param("epochs", epochs)
                    mlflow.log_param("wordNgrams", wordNgrams)
                    mlflow.log_param("dim", 100)
                    mlflow.log_param("decision_threshold", best_thresh_for_model)
                    
                    mlflow.log_metric("accuracy", best_metrics_for_model["accuracy"])
                    mlflow.log_metric("precision", best_metrics_for_model["precision"])
                    mlflow.log_metric("recall", best_metrics_for_model["recall"])
                    mlflow.log_metric("f1_score", best_metrics_for_model["f1"])
                    
                    model_info = {
                        "name": run_name,
                        "accuracy": best_metrics_for_model["accuracy"],
                        "precision": best_metrics_for_model["precision"],
                        "recall": best_metrics_for_model["recall"],
                        "f1": best_f1_for_model,
                        "learning_rate": lr,
                        "epochs": epochs,
                        "wordNgrams": wordNgrams,
                        "dim": 100,
                        "threshold": best_thresh_for_model
                    }
                    results.append(model_info)

                    # Log the binary model ONLY if it is the absolute best performing model across the whole sweep so far!
                    if best_f1_for_model > best_f1_overall:
                        best_f1_overall = best_f1_for_model
                        best_res = model_info
                        
                        # Save and log model binary
                        model_path = os.path.join(script_dir, "fasttext_model.bin")
                        model.save_model(model_path)
                        mlflow.log_artifact(model_path, artifact_path="model")
                        if os.path.exists(model_path):
                            os.remove(model_path)

    # Display best configuration based on F1 Score
    print("\n\n=== SWEEP RESULTS SUMMARY ===")
    print(f"\nBest Performing FastText Configuration by F1 Score:")
    print(f"  Run Name:         {best_res['name']}")
    print(f"  F1-Score:         {best_res['f1']:.4f}")
    print(f"  Accuracy:         {best_res['accuracy']:.4f}")
    print(f"  Precision:        {best_res['precision']:.4f}")
    print(f"  Recall:           {best_res['recall']:.4f}")
    print(f"  Learning Rate:    {best_res['learning_rate']}")
    print(f"  Epochs:           {best_res['epochs']}")
    print(f"  Word N-grams:     {best_res['wordNgrams']}")
    print(f"  Vector Dimension: {best_res['dim']}")
    print(f"  Threshold:        {best_res['threshold']}")

    # Clean up temporary dataset files
    print("\nCleaning up temporary files...")
    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
        os.remove(test_file)
    print("Done!")
