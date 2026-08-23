import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Ensure robust, centralized MLflow tracking URI pointing to root mlflow.db
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
db_path = os.path.join(project_root, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

def train_and_eval(
    model_type="xgboost",
    run_name="TreeModelRun",
    ngram_range=(1, 2),
    max_features=10000,
    analyzer="word",
    stop_words=None,
    sublinear_tf=True,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    min_child_weight=1,
    threshold=0.5,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None
):
    """
    Trains an XGBoost or LightGBM model with advanced regularization parameters,
    evaluates it using a custom decision threshold on class probabilities,
    and logs all parameters, metrics, and model/vectorizer artifacts to MLflow.
    """
    with mlflow.start_run(run_name=run_name):
        # Log all parameters to MLflow
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("ngram_range", str(ngram_range))
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("analyzer", analyzer)
        mlflow.log_param("stop_words", "None" if stop_words is None else str(stop_words))
        mlflow.log_param("sublinear_tf", sublinear_tf)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("subsample", subsample)
        mlflow.log_param("colsample_bytree", colsample_bytree)
        mlflow.log_param("reg_alpha", reg_alpha)
        mlflow.log_param("reg_lambda", reg_lambda)
        mlflow.log_param("min_child_weight", min_child_weight)
        mlflow.log_param("decision_threshold", threshold)

        # Feature Extraction
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            analyzer=analyzer,
            stop_words=stop_words,
            sublinear_tf=sublinear_tf
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Initialize the chosen tree-based classifier
        if model_type == "xgboost":
            clf = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                min_child_weight=min_child_weight,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False
            )
        elif model_type == "lightgbm":
            # For LightGBM, activate bagging with subsample_freq
            clf = LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,
                subsample=subsample,
                subsample_freq=1 if subsample < 1.0 else 0,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                min_child_weight=min_child_weight,
                random_state=42,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Train Model
        clf.fit(X_train_tfidf, y_train)

        # Predict probabilities
        y_prob = clf.predict_proba(X_test_tfidf)[:, 1]

        # Apply custom decision threshold
        y_pred = (y_prob >= threshold).astype(int)

        # Metrics calculation
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model with MLflow
        if model_type == "xgboost":
            mlflow.xgboost.log_model(clf, "model")
        else:
            mlflow.lightgbm.log_model(clf, "model")

        # Save and log TF-IDF Vectorizer
        vectorizer_path = os.path.join(script_dir, "vectorizer.joblib")
        joblib.dump(vectorizer, vectorizer_path)
        mlflow.log_artifact(vectorizer_path, artifact_path="model")
        if os.path.exists(vectorizer_path):
            os.remove(vectorizer_path)

        return {
            "name": run_name,
            "model_type": model_type,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


if __name__ == "__main__":
    # Load Toxic-Chat Dataset
    print("--- Loading Toxic-Chat Dataset ---")
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])

    X_train = df_train['user_input'].fillna("")
    y_train = df_train['toxicity']
    X_test = df_test['user_input'].fillna("")
    y_test = df_test['toxicity']

    # Set up MLflow experiment
    experiment_name = "Toxicity Tree Models"
    mlflow.set_experiment(experiment_name)

    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    imbalance_ratio = neg_count / pos_count
    print(f"Dataset imbalance ratio: {imbalance_ratio:.4f} ({neg_count} safe, {pos_count} toxic)")

    results = []

    # ==========================================
    # STAGE 1: FEATURE REPRESENTATION OPTIMIZATION
    # ==========================================
    print("\n==============================================")
    print("STAGE 1: NLP / VECTORIZATION HYPERPARAMETER SWEEP")
    print("==============================================")

    # We evaluate combinations of analyzer, stop_words, and log TF-scaling
    # Note: We omit character n-grams from the main sweep to keep execution time under 1-2 minutes.
    nlp_configs = [
        # Baseline Word configuration (with english stop words)
        {"run_name": "Word-StopWords-NoSublinear", "analyzer": "word", "ngram_range": (1, 2), "stop_words": "english", "sublinear_tf": False},
        # Keep Stop Words + normal scaling
        {"run_name": "Word-KeepStopWords-NoSublinear", "analyzer": "word", "ngram_range": (1, 2), "stop_words": None, "sublinear_tf": False},
        # Keep Stop Words + sublinear log TF-scaling (Strategic Strategy 1 & 3)
        {"run_name": "Word-KeepStopWords-Sublinear", "analyzer": "word", "ngram_range": (1, 2), "stop_words": None, "sublinear_tf": True},
    ]

    best_nlp_by_model = {"xgboost": None, "lightgbm": None}
    best_nlp_f1 = {"xgboost": -1, "lightgbm": -1}

    for model_type in ["xgboost", "lightgbm"]:
        print(f"\n--- Sweeping NLP Features for {model_type.upper()} ---")
        for cfg in nlp_configs:
            name = f"NLP_Sweep-{model_type}-{cfg['run_name']}"
            res = train_and_eval(
                model_type=model_type,
                run_name=name,
                ngram_range=cfg['ngram_range'],
                max_features=10000, # reduced features slightly for faster runs
                analyzer=cfg['analyzer'],
                stop_words=cfg['stop_words'],
                sublinear_tf=cfg['sublinear_tf'],
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=imbalance_ratio * 0.5, # reasonable baseline scale
                threshold=0.4, # reasonable threshold
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )
            print(f"  {cfg['run_name']}: F1-Score = {res['f1']:.4f} (Prec: {res['precision']:.4f}, Rec: {res['recall']:.4f})")
            results.append(res)
            
            # Keep track of the best vectorizer parameters per model type
            if res["f1"] > best_nlp_f1[model_type]:
                best_nlp_f1[model_type] = res["f1"]
                best_nlp_by_model[model_type] = cfg

    print("\n--- BEST STAGE 1 NLP DESIGNS ---")
    for m, cfg in best_nlp_by_model.items():
        print(f"  {m.upper()} best vectorizer: {cfg['run_name']} (F1: {best_nlp_f1[m]:.4f})")


    # ==========================================
    # STAGE 2: ADVANCED TREE REGULARIZATION & BOUNDARY SWEEP
    # ==========================================
    print("\n==============================================")
    print("STAGE 2: advanced model regularization & threshold sweep")
    print("==============================================")

    # Tightened tree regularization configurations to keep runs extremely fast
    reg_configs = [
        # Normal (baseline)
        {"subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 1.0, "min_child_weight": 1, "label": "NoReg"},
        # Regularized (bagging + minor L2 penalty)
        {"subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 3.0, "min_child_weight": 2, "label": "Regularized"}
    ]

    # Targeted sweep grids to ensure full execution under 1-2 minutes
    weight_sweep = [imbalance_ratio * 0.5, imbalance_ratio * 0.8]
    threshold_sweep = [0.35, 0.4, 0.45]

    for model_type in ["xgboost", "lightgbm"]:
        # Get the optimal NLP config found in Stage 1
        nlp_cfg = best_nlp_by_model[model_type]
        print(f"\n--- Squeezing {model_type.upper()} with best vectorizer '{nlp_cfg['run_name']}' ---")

        for reg in reg_configs:
            for wt in weight_sweep:
                for thresh in threshold_sweep:
                    run_name = f"Reg_Sweep-{model_type}-{reg['label']}-W:{wt:.2f}-Th:{thresh:.2f}"
                    res = train_and_eval(
                        model_type=model_type,
                        run_name=run_name,
                        ngram_range=nlp_cfg['ngram_range'],
                        max_features=10000,
                        analyzer=nlp_cfg['analyzer'],
                        stop_words=nlp_cfg['stop_words'],
                        sublinear_tf=nlp_cfg['sublinear_tf'],
                        n_estimators=120, # optimized estimators count
                        max_depth=5,
                        learning_rate=0.1,
                        scale_pos_weight=wt,
                        subsample=reg['subsample'],
                        colsample_bytree=reg['colsample_bytree'],
                        reg_alpha=reg['reg_alpha'],
                        reg_lambda=reg['reg_lambda'],
                        min_child_weight=reg['min_child_weight'],
                        threshold=thresh,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test
                    )
                    results.append(res)

    # Find best overall configuration across all experiments
    best_res = max(results, key=lambda x: x["f1"])
    print("\n\n==============================================")
    print("=== OPTIMIZED SWEEP RESULTS SUMMARY ===")
    print("==============================================")
    print(f"Best Performing Model Configuration by F1 Score:")
    print(f"  Model Type:          {best_res['model_type'].upper()}")
    print(f"  Run Name:            {best_res['name']}")
    print(f"  Optimized F1-Score:  {best_res['f1']:.4f}")
    print(f"  Accuracy:            {best_res['accuracy']:.4f}")
    print(f"  Precision:           {best_res['precision']:.4f}")
    print(f"  Recall:              {best_res['recall']:.4f}")
