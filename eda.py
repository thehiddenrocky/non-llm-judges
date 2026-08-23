from datasets import load_dataset
import pandas as pd

print("--- Loading Toxic-Chat Dataset for EDA ---")
dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
df = pd.DataFrame(dataset['train'])

print("\n--- Basic Statistics ---")
print(f"Total rows: {len(df)}")
print("\nClass Distribution (0 = Safe, 1 = Toxic):")
print(df['toxicity'].value_counts(normalize=True))
print(df['toxicity'].value_counts())

print("\n--- Sample Toxic Examples (Toxicity = 1) ---")
toxic_examples = df[df['toxicity'] == 1]['model_output'].head(10).tolist()
for i, ex in enumerate(toxic_examples, 1):
    print(f"{i}. {ex[:200]}{'...' if len(ex) > 200 else ''}")

print("\n--- Sample Safe Examples (Toxicity = 0) ---")
safe_examples = df[df['toxicity'] == 0]['model_output'].head(5).tolist()
for i, ex in enumerate(safe_examples, 1):
    print(f"{i}. {ex[:200]}{'...' if len(ex) > 200 else ''}")

print("\n--- Text Length Analysis ---")
df['text_len'] = df['model_output'].str.len()
print("Average length by toxicity:")
print(df.groupby('toxicity')['text_len'].mean())

# Look for specific keywords in toxic vs non-toxic
from collections import Counter
import re

def get_top_words(texts, n=20):
    words = []
    for t in texts:
        # Simple tokenization
        words.extend(re.findall(r'\w+', str(t).lower()))
    return Counter(words).most_common(n)

print("\n--- Common Words in Toxic Outputs ---")
print(get_top_words(df[df['toxicity'] == 1]['model_output']))

print("\n--- Common Words in Safe Outputs ---")
print(get_top_words(df[df['toxicity'] == 0]['model_output']))
