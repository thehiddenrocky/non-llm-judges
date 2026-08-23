# MLOps Transformer Experiments

This directory contains MLOps experiments to improve the toxicity classification F1-Score using Encoder-Only Transformers (BERT-family models).

By leveraging deep contextual representations, these models capture complex semantic nuances (such as passive-aggressive comments, jailbreak prompts, or sarcasm) that traditional bag-of-words approaches (like TF-IDF + SVM) miss.

## Features
- **Centralized MLOps Integration**: Pointed to the central SQLite tracking URI (`mlflow.db` in project root) to compare traditional SVM/Tree models and state-of-the-art Transformer models side-by-side in the same MLflow dashboard.
- **Robust Out-of-the-box Evaluation**: Evaluate pre-trained binary toxicity classifiers on the LMSYS Toxic-Chat test set directly.
- **Efficient Fine-Tuning**: Support fine-tuning of small encoder-only models on the Toxic-Chat training set, with automatic device routing (Metal MPS on macOS, CUDA on Windows/Linux, or CPU).
- **Class-Imbalance Handling**: Apply dynamic class-imbalance weights in the cross-entropy loss based on the training dataset's label distribution.
- **Decision Threshold Sweep**: Sweep classification probability thresholds from `0.1` to `0.9` during validation to find the decision boundary that maximizes the **F1-Score**.
- **Comprehensive Logging**: Auto-logs parameters (model name, LR, batch size, weight decay, epochs, etc.), evaluation metrics across all thresholds, and logs the fine-tuned model checkpoint as a logged artifact.

---

## Setup & Dependencies

Make sure the required dependencies are installed in the virtual environment. From the repository root, run:

```bash
# If not already installed:
./venv/bin/pip install torch transformers accelerate evaluate scikit-learn pandas datasets mlflow
```

---

## Running Experiments

All commands should be executed from the repository root.

### 1. Evaluate an Out-of-the-Box Pretrained Model
Evaluate a pre-trained toxicity classifier directly on the `toxic-chat` test set:

```bash
./venv/bin/python mlops/train_transformer.py \
  --mode evaluate-pretrained \
  --model_name valhalla/distilbert-toxicity-classifier \
  --batch_size 32
```

### 2. Fine-Tune a Fast, Tiny BERT Model (Fast Validation Run)
To verify your end-to-end training pipeline quickly (takes ~15-30 seconds on CPU):

```bash
./venv/bin/python mlops/train_transformer.py \
  --mode fine-tune \
  --model_name prajjwal1/bert-tiny \
  --epochs 2 \
  --batch_size 64
```

### 3. Fine-Tune a MiniLM Classifier (Highly Recommended - Sub-10ms Latency)
A lightweight model (`sentence-transformers/all-MiniLM-L6-v2`) that has an exceptional accuracy-latency tradeoff:

```bash
./venv/bin/python mlops/train_transformer.py \
  --mode fine-tune \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --epochs 3 \
  --batch_size 32 \
  --lr 3e-5 \
  --use_class_weights
```

### 4. Fine-Tune DistilBERT Classifier
A larger DistilBERT model for standard benchmarking:

```bash
./venv/bin/python mlops/train_transformer.py \
  --mode fine-tune \
  --model_name distilbert-base-uncased \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --use_class_weights
```

---

## Comparing Models with MLflow

To launch the unified MLflow Dashboard and view SVM, Tree Models, and Transformer experiments side-by-side, run:

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser. You can:
1. Select the `Toxicity Transformers` experiment to compare pre-trained models versus fine-tuned models.
2. Select all experiments to see how the best SVM, XGBoost, LightGBM, and DistilBERT models compare on key metrics like F1-Score and Recall.
