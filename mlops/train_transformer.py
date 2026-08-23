import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.pytorch
import shutil

# Robust, centralized MLflow tracking URI pointing to root mlflow.db
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
db_path = os.path.join(project_root, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Expected to be a tensor of weights for each class
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            focal_loss = focal_loss * at

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ToxicChatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Binary classification probabilities
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def run_evaluation_sweep(y_true, y_prob):
    """
    Sweeps decision thresholds to find the threshold that maximizes F1 score.
    """
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_f1 = -1.0
    best_thresh = 0.5
    best_metrics = {}

    results = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        results.append({
            "threshold": thresh,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

    return best_thresh, best_metrics, results

def fine_tune(args, X_train, y_train, X_test, y_test):
    # Set up MLflow experiment
    mlflow.set_experiment("Toxicity Transformers")

    run_name = f"FineTune - {args.model_name.split('/')[-1]} - LR:{args.lr} - Ep:{args.epochs}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n--- Starting Fine-Tuning: {run_name} ---")
        
        # Log training parameters
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("weight_decay", args.weight_decay)
        mlflow.log_param("max_len", args.max_len)
        mlflow.log_param("use_class_weights", args.use_class_weights)
        mlflow.log_param("use_focal_loss", getattr(args, "use_focal_loss", False))

        # Device configuration
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        mlflow.log_param("device", str(device))

        # Tokenizer & Model
        print(f"Loading tokenizer & model: {args.model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        except Exception as e:
            print(f"AutoTokenizer failed with: {e}. Retrying with BertTokenizer...")
            try:
                from transformers import BertTokenizer
                tokenizer = BertTokenizer.from_pretrained(args.model_name)
            except Exception as e2:
                print(f"BertTokenizer fallback failed: {e2}. Attempting AutoTokenizer with use_fast=False...")
                tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
        model.to(device)

        # Datasets & Dataloaders
        train_dataset = ToxicChatDataset(X_train.tolist(), y_train.tolist(), tokenizer, args.max_len)
        test_dataset = ToxicChatDataset(X_test.tolist(), y_test.tolist(), tokenizer, args.max_len)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Class Weights for Loss
        weight_tensor = None
        if args.use_class_weights:
            neg_count = sum(y_train == 0)
            pos_count = sum(y_train == 1)
            imbalance_ratio = neg_count / pos_count
            print(f"Applying class weights (pos_weight={imbalance_ratio:.2f})")
            weight_tensor = torch.tensor([1.0, imbalance_ratio], dtype=torch.float, device=device)
            
        if getattr(args, "use_focal_loss", False):
            print("Using Focal Loss")
            loss_fn = FocalLoss(alpha=weight_tensor, gamma=2.0)
        else:
            if weight_tensor is not None:
                loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
            else:
                loss_fn = nn.CrossEntropyLoss()

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Training Loop
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            print(f"Epoch {epoch+1}/{args.epochs}")
            
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = loss_fn(logits, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                if (step + 1) % 50 == 0 or step == len(train_loader) - 1:
                    print(f"  Step {step+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Evaluation
        print("Evaluating fine-tuned model on test set...")
        y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)

        # Standard Evaluation (Threshold = 0.5)
        acc_05 = accuracy_score(y_true, y_pred)
        p_05, r_05, f1_05, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        mlflow.log_metric("accuracy_t0.5", acc_05)
        mlflow.log_metric("precision_t0.5", p_05)
        mlflow.log_metric("recall_t0.5", r_05)
        mlflow.log_metric("f1_score_t0.5", f1_05)

        print("\nClassification Report (Threshold = 0.5):")
        print(classification_report(y_true, y_pred, zero_division=0))

        # Sweep Decision Thresholds
        best_thresh, best_metrics, sweep_results = run_evaluation_sweep(y_true, y_prob)
        print(f"\nOptimal Decision Threshold identified: {best_thresh}")
        print(f"Best Metrics: F1={best_metrics['f1']:.4f}, Accuracy={best_metrics['accuracy']:.4f}, Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}")

        # Log Sweep Metrics
        mlflow.log_param("best_threshold", best_thresh)
        mlflow.log_metric("accuracy", best_metrics["accuracy"])
        mlflow.log_metric("precision", best_metrics["precision"])
        mlflow.log_metric("recall", best_metrics["recall"])
        mlflow.log_metric("f1_score", best_metrics["f1"])

        for item in sweep_results:
            t = item["threshold"]
            mlflow.log_metric(f"sweep_f1_t{t:.1f}", item["f1"])
            mlflow.log_metric(f"sweep_recall_t{t:.1f}", item["recall"])

        # Save model locally and log as MLflow artifact
        local_model_dir = os.path.join(script_dir, "fine_tuned_model")
        if os.path.exists(local_model_dir):
            shutil.rmtree(local_model_dir)
        os.makedirs(local_model_dir)

        model.save_pretrained(local_model_dir)
        tokenizer.save_pretrained(local_model_dir)
        
        # Log model artifacts
        mlflow.log_artifacts(local_model_dir, artifact_path="model")
        
        # Cleanup local model directory
        shutil.rmtree(local_model_dir)

        return {
            "name": run_name,
            "best_threshold": best_thresh,
            **best_metrics
        }

def evaluate_pretrained(args, X_test, y_test):
    # Set up MLflow experiment
    mlflow.set_experiment("Toxicity Transformers")

    run_name = f"PretrainedEval - {args.model_name.split('/')[-1]}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n--- Starting Pretrained Evaluation: {run_name} ---")
        
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("max_len", args.max_len)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        mlflow.log_param("device", str(device))

        print(f"Loading pre-trained tokenizer & model: {args.model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        except Exception as e:
            print(f"AutoTokenizer failed with: {e}. Retrying with BertTokenizer...")
            try:
                from transformers import BertTokenizer
                tokenizer = BertTokenizer.from_pretrained(args.model_name)
            except Exception as e2:
                print(f"BertTokenizer fallback failed: {e2}. Attempting AutoTokenizer with use_fast=False...")
                tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
        model.to(device)

        test_dataset = ToxicChatDataset(X_test.tolist(), y_test.tolist(), tokenizer, args.max_len)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print("Evaluating pre-trained model on test set...")
        y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)

        # Standard metrics
        acc_05 = accuracy_score(y_true, y_pred)
        p_05, r_05, f1_05, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        mlflow.log_metric("accuracy_t0.5", acc_05)
        mlflow.log_metric("precision_t0.5", p_05)
        mlflow.log_metric("recall_t0.5", r_05)
        mlflow.log_metric("f1_score_t0.5", f1_05)

        print("\nClassification Report (Threshold = 0.5):")
        print(classification_report(y_true, y_pred, zero_division=0))

        # Sweep decision thresholds
        best_thresh, best_metrics, sweep_results = run_evaluation_sweep(y_true, y_prob)
        print(f"\nOptimal Decision Threshold identified: {best_thresh}")
        print(f"Best Metrics: F1={best_metrics['f1']:.4f}, Accuracy={best_metrics['accuracy']:.4f}, Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}")

        # Log Sweep Metrics
        mlflow.log_param("best_threshold", best_thresh)
        mlflow.log_metric("accuracy", best_metrics["accuracy"])
        mlflow.log_metric("precision", best_metrics["precision"])
        mlflow.log_metric("recall", best_metrics["recall"])
        mlflow.log_metric("f1_score", best_metrics["f1"])

        return {
            "name": run_name,
            "best_threshold": best_thresh,
            **best_metrics
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps toxicity classification experiments using Transformer models.")
    parser.add_argument("--mode", type=str, default="fine-tune", choices=["fine-tune", "evaluate-pretrained"],
                        help="Execution mode: 'fine-tune' a model or 'evaluate-pretrained' directly.")
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-tiny",
                        help="Pretrained Hugging Face model checkpoint (e.g., prajjwal1/bert-tiny, sentence-transformers/all-MiniLM-L6-v2, distilbert-base-uncased, valhalla/distilbert-toxicity-classifier).")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimization.")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum tokenization sequence length.")
    parser.add_argument("--use_class_weights", action="store_true", help="Apply class weights in loss function to handle imbalance.")
    parser.add_argument("--use_focal_loss", action="store_true", help="Use Focal Loss instead of Cross Entropy to focus on hard examples.")

    args = parser.parse_args()

    # Load dataset
    print("--- Loading Toxic-Chat Dataset ---")
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])

    X_train = df_train['user_input'].fillna("")
    y_train = df_train['toxicity']
    X_test = df_test['user_input'].fillna("")
    y_test = df_test['toxicity']

    if args.mode == "fine-tune":
        result = fine_tune(args, X_train, y_train, X_test, y_test)
    else:
        result = evaluate_pretrained(args, X_test, y_test)

    print("\n=== EXPERIMENT COMPLETED ===")
    print(f"Run Name: {result['name']}")
    print(f"Best Threshold: {result['best_threshold']}")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1 Score:  {result['f1']:.4f}")
