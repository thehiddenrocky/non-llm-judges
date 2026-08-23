from datasets import load_dataset
import pandas as pd

dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
df_train = pd.DataFrame(dataset['train'])

print("Columns in dataset:")
print(df_train.columns.tolist())

print("\nFirst 5 rows of dataset:")
pd.set_option('display.max_columns', None)
print(df_train.head(3))

print("\nLet's check toxicity correlation of user_input vs model_output:")
# If there are separate labels for user and model output:
# Let's inspect some toxic rows where toxicity == 1
toxic_rows = df_train[df_train['toxicity'] == 1].head(10)
for idx, row in toxic_rows.iterrows():
    print(f"\n--- Row {idx} ---")
    print(f"User Input: {repr(row['user_input'])[:300]}")
    print(f"Model Output: {repr(row['model_output'])[:300]}")
    print(f"Toxicity Label: {row['toxicity']}")
    if 'user_toxicity' in row:
        print(f"User Toxicity: {row['user_toxicity']}")
    if 'model_toxicity' in row:
        print(f"Model Toxicity: {row['model_toxicity']}")
