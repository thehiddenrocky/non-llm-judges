from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

print("--- Loading Toxic-Chat Dataset ---")
dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
print("Extracting splits...")
df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])

X_train = df_train['model_output'].fillna("")
y_train = df_train['toxicity']
X_test = df_test['model_output'].fillna("")
y_test = df_test['toxicity']

print(f"Data statistics:")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Testing samples:  {len(X_test)}")
print(f"  - Toxicity rate (train): {y_train.mean():.2%}")

print("\n--- Feature Extraction ---")
print("Vectorizing text using TF-IDF (max_features=10000)...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

print("\n--- Model Training ---")
print("Fitting Linear SVM classifier...")
clf = LinearSVC(random_state=42, tol=1e-5)
clf.fit(X_train_tfidf, y_train)
print("Training complete.")

print("\n--- Evaluation ---")
print("Generating predictions on test set...")
y_pred = clf.predict(X_test_tfidf)

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
