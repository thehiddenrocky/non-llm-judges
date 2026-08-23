from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

print("Loading dataset...")
# 1. Load the dataset
dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])

# 2. Focus on the relevant columns
# 'user_input' or 'model_output' could be evaluated. 
# The previous snippet used 'model_output'.
X_train = df_train['model_output'].fillna("")
y_train = df_train['toxicity']
X_test = df_test['model_output'].fillna("")
y_test = df_test['toxicity']

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 3. Vectorization
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train Linear SVM
print("Training Linear SVM classifier...")
clf = LinearSVC(random_state=42, tol=1e-5)
clf.fit(X_train_tfidf, y_train)

# 5. Evaluation
print("Evaluating model...")
y_pred = clf.predict(X_test_tfidf)

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
