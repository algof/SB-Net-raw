import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import gc

print("Starting combined data processing and encoding script [Sensor 1-3 Standalone - Target Dtypes]")

output_dir = 'final_dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

dataset_files = {
    '1': 'sensor1/sensor1.binetflow',
    '2': 'sensor2/sensor2.binetflow',
    '3': 'sensor3/sensor3.binetflow',
}

df_list = []
loaded_keys = []
all_original_data = []

print("Loading all raw sensor dataframes for encoder fitting...")
for key, path in dataset_files.items():
    try:
        dtype_spec = {'sTos': str, 'dTos': str}
        df = pd.read_csv(path, dtype=dtype_spec)
        if 'dTos' in df.columns:
            df['dTos'] = pd.to_numeric(df['dTos'], errors='coerce').fillna(0)
        if 'sTos' in df.columns:
            df['sTos'] = pd.to_numeric(df['sTos'], errors='coerce').fillna(0)
        df_list.append(df)
        all_original_data.append(df.copy())
        loaded_keys.append(key)
        print(f"Successfully loaded: {path} (key: {key}, shape: {df.shape})")
    except FileNotFoundError:
        print(f"Warning: Dataset file not found at {path}. Skipping.")
    except Exception as e:
        print(f"Warning: Could not load {path}. Error: {e}. Skipping.")

if not df_list:
    print("Error: No .binetflow datasets could be loaded. Stopping script.")
    exit()

print("Fitting LabelEncoders on combined raw sensor data...")
combined_fit_df = pd.concat(all_original_data, ignore_index=True)
del all_original_data
gc.collect()

categorical_cols = ['SrcAddr', 'Sport', 'Dir', 'DstAddr', 'Dport', 'State']
encoders = {}
imputation_values = {}

for col in categorical_cols:
    if col in combined_fit_df.columns:
        print(f"Fitting LabelEncoder for: {col}")
        combined_fit_df[col] = combined_fit_df[col].fillna('nan').astype(str)
        unique_values = combined_fit_df[col].unique()
        le = LabelEncoder()
        le.fit(unique_values)
        encoders[col] = le
        if 'nan' in le.classes_:
            imputation_values[col] = int(le.transform(['nan'])[0])
            print(f"  Found NaN, its encoded ID is: {imputation_values[col]}")
        else:
             imputation_values[col] = 0
             print(f"  No NaN found, using fallback ID: {imputation_values[col]}")
    else:
        print(f"Warning: Column '{col}' not found in combined data. Skipping encoding for it.")

print("Encoders are ready.")
del combined_fit_df
gc.collect()

def categorize_label(label):
    label_str = str(label).lower()
    if 'botnet' in label_str:
        if 'spam' in label_str: return 'botnet_spam'
        else: return 'botnet'
    elif 'background' in label_str or 'normal' in label_str: return 'normal'
    else: return 'normal'

for i in range(len(df_list)):
    df_list[i]['Label'] = df_list[i]['Label'].apply(categorize_label)
print("Applied label simplification.")

columns_to_drop = ['dTos', 'sTos', 'ActivityLabel', 'BotnetName', 'SensorId', 'StartTime']
for i in range(len(df_list)):
    cols_to_drop_existing = [col for col in columns_to_drop if col in df_list[i].columns]
    if cols_to_drop_existing:
        df_list[i] = df_list[i].drop(columns=cols_to_drop_existing, errors='ignore')
print(f"Dropped unnecessary columns (if they existed).")

normal_train, normal_test = [], []
botnet_train, botnet_test = [], []
botnet_spam_train, botnet_spam_test = [], []
normal_df, botnet_df, botnet_spam_df = [], [], []

for df in df_list:
    normal_df.append(df[df['Label'] == 'normal'])
    botnet_df.append(df[df['Label'] == 'botnet'])
    botnet_spam_df.append(df[df['Label'] == 'botnet_spam'])
del df_list
gc.collect()

for i in range(len(loaded_keys)):
    if i < len(normal_df) and not normal_df[i].empty:
        tr, te = train_test_split(normal_df[i], test_size=0.3, random_state=42)
        normal_train.append(tr); normal_test.append(te)
    else:
        normal_train.append(pd.DataFrame()); normal_test.append(pd.DataFrame())
    if i < len(botnet_df) and not botnet_df[i].empty:
        tr, te = train_test_split(botnet_df[i], test_size=0.3, random_state=42)
        botnet_train.append(tr); botnet_test.append(te)
    else:
        botnet_train.append(pd.DataFrame()); botnet_test.append(pd.DataFrame())
    if i < len(botnet_spam_df) and not botnet_spam_df[i].empty:
        tr, te = train_test_split(botnet_spam_df[i], test_size=0.3, random_state=42)
        botnet_spam_train.append(tr); botnet_spam_test.append(te)
    else:
        botnet_spam_train.append(pd.DataFrame()); botnet_spam_test.append(pd.DataFrame())

print("Performed stratified train-test split.")
del normal_df, botnet_df, botnet_spam_df
gc.collect()

temp_train_df = []
num_files = len(loaded_keys)
for i in range(num_files):
    file_dfs = []
    if i < len(normal_train) and not normal_train[i].empty: file_dfs.append(normal_train[i])
    if i < len(botnet_train) and not botnet_train[i].empty: file_dfs.append(botnet_train[i])
    if i < len(botnet_spam_train) and not botnet_spam_train[i].empty: file_dfs.append(botnet_spam_train[i])
    if file_dfs:
        temp_train_df.append(pd.concat(file_dfs, ignore_index=True))
del normal_train, botnet_train, botnet_spam_train
gc.collect()

if not temp_train_df:
     print("Error: No training data was generated."); exit()

train_df = pd.concat(temp_train_df, ignore_index=True)
del temp_train_df
gc.collect()
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Combined training data (shape: {train_df.shape})")

test_dfs = {}
for i in range(num_files):
    file_dfs = []
    if i < len(normal_test) and not normal_test[i].empty: file_dfs.append(normal_test[i])
    if i < len(botnet_test) and not botnet_test[i].empty: file_dfs.append(botnet_test[i])
    if i < len(botnet_spam_test) and not botnet_spam_test[i].empty: file_dfs.append(botnet_spam_test[i])
    if file_dfs:
        df = pd.concat(file_dfs, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        test_dfs[loaded_keys[i]] = df
del normal_test, botnet_test, botnet_spam_test
gc.collect()
print(f"Combined test data (number of test sets: {len(test_dfs)})")

all_dfs_to_transform = [train_df] + list(test_dfs.values())
del train_df
gc.collect()

final_dfs_list = []

print("Applying transformations (Proto dummies, fitted Encoders) to all datasets...")
for df_orig in all_dfs_to_transform:
    df = df_orig.copy()
    print(f"Processing DataFrame partition (Initial shape: {df.shape})")

    if 'Proto' in df.columns:
        df = pd.get_dummies(df, columns=['Proto'], prefix='', prefix_sep='')
        print(f"  Shape after get_dummies: {df.shape}")

    for col in categorical_cols:
         if col in encoders:
             df[col] = df[col].fillna('nan').astype(str)
             known_mask = df[col].isin(encoders[col].classes_)
             known_indices = df.index[known_mask]
             unknown_indices = df.index[~known_mask]
             if not known_indices.empty:
                  df.loc[known_indices, col] = encoders[col].transform(df.loc[known_indices, col])
             unknown_fill_value = imputation_values.get(col, 0)
             if not unknown_indices.empty:
                  df.loc[unknown_indices, col] = unknown_fill_value
         elif col in df.columns:
             df[col] = imputation_values.get(col, 0)

    protocol_cols_from_target = [
        'arp', 'esp', 'gre', 'icmp', 'igmp', 'ipv6', 'ipv6-icmp', 'ipx/spx', 'llc',
        'pim', 'rarp', 'rsvp', 'rtcp', 'rtp', 'tcp', 'udp', 'udt', 'unas', 'ipnip'
    ]
    core_cols = ['Dur', 'SrcAddr', 'Sport', 'Dir', 'DstAddr', 'Dport', 'State',
                 'TotPkts', 'TotBytes', 'SrcBytes', 'Label']
    final_column_order = core_cols + protocol_cols_from_target

    for col in final_column_order:
        if col not in df.columns:
            df[col] = 0

    df_final = df[final_column_order].copy()

    print(f"  Applying target data types...")
    for col in df_final.columns:
        if col == 'Dur':
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0).astype(np.float64)
        elif col == 'Label':
            df_final[col] = df_final[col].astype(object)
        else:
             df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0).astype(np.int64)

    print(f"  Shape after finalizing columns and types: {df_final.shape}")
    final_dfs_list.append(df_final)
    del df, df_orig, df_final
    gc.collect()

print("Transformations complete.")

final_train_df = final_dfs_list[0]
final_test_dfs = {key: df for key, df in zip(test_dfs.keys(), final_dfs_list[1:])}

train_output_file = os.path.join(output_dir, 'train.csv')
final_train_df.to_csv(train_output_file, index=False)
print(f"Saved ENCODED training data to {train_output_file} (shape: {final_train_df.shape})")

for key, df in final_test_dfs.items():
    test_output_file = os.path.join(output_dir, f"test_{key}.csv")
    df.to_csv(test_output_file, index=False)
    print(f"Saved ENCODED test data to {test_output_file} (shape: {df.shape})")

print("\n--- All-in-one processing and encoding complete for Sensor 1-3! ---")
print(f"Final files are in the '{output_dir}' directory.")