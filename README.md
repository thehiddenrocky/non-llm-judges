# Toxic-SVM: Non-LLM Toxicity Judge

A lightweight, high-performance toxicity classifier for chat datasets using Support Vector Machines (SVM) and Tree-Based Ensembles (XGBoost & LightGBM). This project serves as a non-LLM "judge" to evaluate the safety of model outputs from the `lmsys/toxic-chat` dataset.

## Overview

Traditional LLM-based judges are computationally expensive. This project demonstrates that a classical machine learning approach (TF-IDF + Linear SVM/Trees) can achieve high accuracy (93%+) on toxicity detection tasks with minimal latency and resource usage.

## Dataset

Used the **LMSYS Toxic-Chat** dataset (`toxicchat0124` configuration), which contains real-world user-model interactions labeled for toxicity.

## Setup

1. **System Dependencies (macOS only):**
   XGBoost and LightGBM require OpenMP (`libomp`).
   ```bash
   brew install libomp
   ```

2. **Environment:**
   Ensure you have a Python 3.14+ environment.
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Training & Evaluation:**
   * Run the SVM pipeline:
     ```bash
     python train_svm.py
     ```
   * Run the Tree-Based Models (XGBoost & LightGBM) pipeline:
     ```bash
     python tree_models/train_tree.py
     ```

4. **Testing:**
   Run the tree models' unit tests:
   ```bash
   python -m unittest tree_models/test_tree.py
   ```

## Results

Current performance on the `toxicchat0124` test set:

| Model | Configuration | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear SVM** | Baseline | 95.5% | 0.81 | 0.48 | 0.60 |
| **XGBoost** | Baseline (W:1.00, Thresh:0.5) | 94.5% | 0.70 | 0.39 | 0.50 |
| **XGBoost** | Sweep Best (W:6.12, Thresh:0.5) | 93.9% | 0.58 | 0.57 | 0.57 |
| **LightGBM** | Baseline (W:1.00, Thresh:0.5) | 94.4% | 0.69 | 0.38 | 0.49 |
| **LightGBM** | Sweep Best (W:6.12, Thresh:0.5) | 93.7% | 0.56 | 0.57 | 0.56 |

### Findings
* **Class Imbalance & Recall:** The dataset is highly imbalanced (~7.6% toxic). While SVM achieves high accuracy and high precision, its recall is low (0.48).
* **Tree Model Advantage:** By training XGBoost and LightGBM with custom scale positive weights (`scale_pos_weight`) and tuning the decision threshold on predicted probabilities, we boosted recall significantly up to **0.57**, providing a much more robust shield against false negatives at a very minor trade-off in accuracy.

## License
MIT

