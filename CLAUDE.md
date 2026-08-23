# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Toxic-SVM is lightweight toxicity classifier for chat datasets using classical ML (TF-IDF + Linear SVM). Non-LLM judge for evaluating safety of model outputs from the `lmsys/toxic-chat` dataset.

Current performance: 95.5% accuracy on toxicchat0124 test set, though recall for toxic class is lower (0.48), indicating false negatives.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run EDA (exploratory data analysis)
python eda.py

# Train SVM with MLflow experiment tracking
python train_svm.py

# View MLflow dashboard to compare experiments
mlflow ui
```

## Architecture & Core Concepts

### Data Pipeline
- Loads `lmsys/toxic-chat` dataset (`toxicchat0124` configuration) via HuggingFace datasets
- Splits into train/test with labels: 0 = safe, 1 = toxic
- Inputs: `model_output` (text), Labels: `toxicity` (binary)

### Model Architecture
- **Feature Extraction**: TF-IDF vectorizer (max 10k-20k features, 1-2 grams)
- **Classifier**: scikit-learn LinearSVC (linear kernel)
- **Decision Threshold**: Customizable (default 0.0, can shift via decision_function scores for precision/recall tradeoff)

### Experiment Tracking
- **Tool**: MLflow with SQLite backend (`mlflow.db`)
- **Experiment**: "Toxicity SVM"
- **Two baseline runs in train_svm.py**:
  1. Base Model: 1-gram, unbalanced weights, threshold=0
  2. Improved Model: 1-2 grams, balanced class weights, threshold=-0.2 (boosts recall)
- Logged artifacts: parameters, metrics (accuracy/precision/recall/f1), model binaries

### Key Parameters to Tune
- `ngram_range`: (1,1) vs (1,2) for unigrams+bigrams
- `class_weight`: `None` (default) vs `'balanced'` (helps imbalanced toxicity data)
- `threshold`: Shift decision boundary via `decision_function()` to trade precision/recall
- `max_features`: TF-IDF vocab size (10k baseline, 20k for richer vocab)

## Known Issues & Trade-offs

- **Class imbalance**: Toxic samples are minority. Low recall (0.48) on toxic class means false negatives common.
- **Threshold tuning**: Default threshold=0 optimizes for accuracy. Threshold=-0.2 improves recall but may lower precision.
- **Feature engineering**: Static TF-IDF; consider subword tokenization (BPE) or fastText embeddings for better rare-word handling.
