# Plan: MLOps Integration with MLflow

## Objective
Implement experiment tracking and model documentation using MLflow to capture the evolution of the toxicity judge from a baseline to an improved version.

## Key Files & Context
- `train_svm.py`: Main script to be updated with MLflow tracking.
- `requirements.txt`: To be updated with the `mlflow` dependency.

## Implementation Steps

### 1. Environment Preparation
- Install `mlflow` in the virtual environment.
- Update `requirements.txt`.

### 2. Update `train_svm.py`
- Import `mlflow` and `mlflow.sklearn`.
- Refactor the code to define a `run_experiment` function that takes parameters like `ngram_range`, `class_weight`, and `threshold`.
- Wrap the training and evaluation logic within `mlflow.start_run()`:
    - Log parameters: `ngram_range`, `class_weight`, `C`, `threshold`.
    - Log metrics: `accuracy`, `precision`, `recall`, `f1_score`.
    - Log the SVM model and TF-IDF vectorizer as artifacts.
- Run the experiment twice:
    - **Run 1: Base Model** (1-gram, no class weights, threshold 0).
    - **Run 2: Improved Model** (1-2 grams, balanced weights, threshold -0.2).

### 3. Documentation & Visualization
- Provide instructions to launch the MLflow UI (`mlflow ui`) to compare the two runs side-by-side.

## Verification & Testing
- Execute `python train_svm.py` and verify that the `mlruns` directory is created.
- Confirm that metrics for both runs are correctly logged in the console and the MLflow dashboard.
