# Toxic-SVM: Non-LLM Toxicity Judge

A lightweight, high-performance toxicity classifier for chat datasets using Support Vector Machines (SVM). This project serves as a non-LLM "judge" to evaluate the safety of model outputs from the `lmsys/toxic-chat` dataset.

## Overview

Traditional LLM-based judges are computationally expensive. This project demonstrates that a classical machine learning approach (TF-IDF + Linear SVM) can achieve high accuracy (95%+) on toxicity detection tasks with minimal latency and resource usage.

## Dataset

Used the **LMSYS Toxic-Chat** dataset (`toxicchat0124` configuration), which contains real-world user-model interactions labeled for toxicity.

## Setup

1. **Environment:**
   Ensure you have a Python 3.14+ environment.
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install datasets pandas scikit-learn
   ```

2. **Training & Evaluation:**
   Run the main script to load the data, train the SVM, and output the performance metrics.
   ```bash
   python train_svm.py
   ```

## Results

Current performance on the `toxicchat0124` test set:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 95.5% |
| **Precision (Toxic)** | 0.81 |
| **Recall (Toxic)** | 0.48 |
| **F1-Score (Toxic)** | 0.60 |

## License
MIT
