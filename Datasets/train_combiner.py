import pandas as pd

# Load datasets
df_ctu = pd.read_csv('CTU-13/final_dataset/train.csv')
df_ncc = pd.read_csv('NCC-2/final_dataset/train.csv')
df_ncc2 = pd.read_csv('NCC/final_dataset/train.csv')

# Get all unique features across datasets
all_features = set(df_ctu.columns) | set(df_ncc.columns) | set(df_ncc2.columns)

# Ensure all DataFrames have the same columns, filling missing ones with 0
df_ctu = df_ctu.reindex(columns=all_features, fill_value=0)
df_ncc = df_ncc.reindex(columns=all_features, fill_value=0)
df_ncc2 = df_ncc2.reindex(columns=all_features, fill_value=0)

# Combine all DataFrames
df_combined = pd.concat([df_ctu, df_ncc, df_ncc2], ignore_index=True)

# Save or use the combined dataset
df_combined.to_csv('combined_train.csv', index=False)

print("Combined dataset shape:", df_combined.shape)