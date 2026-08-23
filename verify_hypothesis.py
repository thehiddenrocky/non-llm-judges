from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

print("--- Loading Toxic-Chat Dataset ---")
dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])

print(f"Train set size: {len(df_train)}, Test set size: {len(df_test)}")
print(f"Toxicity distribution in test set:\n{df_test['toxicity'].value_counts()}")

# --- 1. Current setup: Features = model_output ---
print("\n==============================================")
print("EXPERIMENT 1: Using 'model_output' (Chatbot Response)")
print("==============================================")
X_train_out = df_train['model_output'].fillna("")
X_test_out = df_test['model_output'].fillna("")
y_train = df_train['toxicity']
y_test = df_test['toxicity']

vectorizer_out = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf_out = vectorizer_out.fit_transform(X_train_out)
X_test_tfidf_out = vectorizer_out.transform(X_test_out)

clf_out = LinearSVC(random_state=42, class_weight='balanced')
clf_out.fit(X_train_tfidf_out, y_train)
y_pred_out = clf_out.predict(X_test_tfidf_out)

acc_out = accuracy_score(y_test, y_pred_out)
p_out, r_out, f1_out, _ = precision_recall_fscore_support(y_test, y_pred_out, average='binary')

print(f"Accuracy:  {acc_out:.4f}")
print(f"Precision: {p_out:.4f}")
print(f"Recall:    {r_out:.4f}")
print(f"F1-Score:  {f1_out:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_out))

# --- 2. Proposed setup: Features = user_input ---
print("\n==============================================")
print("EXPERIMENT 2: Using 'user_input' (User Prompt)")
print("==============================================")
X_train_in = df_train['user_input'].fillna("")
X_test_in = df_test['user_input'].fillna("")

vectorizer_in = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf_in = vectorizer_in.fit_transform(X_train_in)
X_test_tfidf_in = vectorizer_in.transform(X_test_in)

clf_in = LinearSVC(random_state=42, class_weight='balanced')
clf_in.fit(X_train_tfidf_in, y_train)
y_pred_in = clf_in.predict(X_test_tfidf_in)

acc_in = accuracy_score(y_test, y_pred_in)
p_in, r_in, f1_in, _ = precision_recall_fscore_support(y_test, y_pred_in, average='binary')

print(f"Accuracy:  {acc_in:.4f}")
print(f"Precision: {p_in:.4f}")
print(f"Recall:    {r_in:.4f}")
print(f"F1-Score:  {f1_in:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_in))
