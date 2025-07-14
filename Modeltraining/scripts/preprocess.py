import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load your JSON data
with open('Modeltraining/data/mcq_dataset_90.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv('Modeltraining/data/mcq_dataset_90.csv', index=False)


# Split the data (80% train, 20% validation)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save splits
train_df.to_csv('Modeltraining/data/mcq_train.csv', index=False)
val_df.to_csv('Modeltraining/data/mcq_val.csv', index=False)