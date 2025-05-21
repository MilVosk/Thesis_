import pandas as pd
from sklearn.metrics import f1_score

# Load the CSV file
df = pd.read_csv('predicted_relations.csv')

# Extract the true and predicted labels
y_true = df['gold label']
y_pred = df['Relation_Present']

# Calculate the micro-averaged F1 score
f1_micro = f1_score(y_true, y_pred, average='micro')

# Print the result
print(f"Micro-averaged F1 score: {f1_micro:.4f}")
