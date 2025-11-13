import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import time

# Define models
models = {
    'Random Forest (Gini)': RandomForestClassifier(criterion='gini', max_depth=29, random_state=42),
}

selected_feature_cols = [
    'SrcAddr',   # 0
    'Sport',     # 1
    'Dport',     # 2
    # 'DstAddr',   # 3
    # 'SrcBytes',  # 4
    # 'TotBytes',  # 5
    # 'State',     # 6
    # 'Dur',       # 7
    # 'tcp',       # 8
    # 'TotPkts',   # 9
    # 'Dir',       # 10
    # 'udp',       # 11
    # 'rtp',       # 12
    # 'icmp',      # 13
    # 'igmp',      # 14
    # 'rtcp',      # 15
    # 'ipx/spx',   # 16
    # 'ipv6-icmp',  # 17
    # 'arp',       # 18
    # 'pim',       # 19
    # 'rarp',      # 20
    # 'esp',       # 21
    # 'ipv6',      # 22
    # 'llc',       # 23
    # 'gre',       # 24
    # 'unas',      # 25
    # 'udt',       # 26
    # 'ipnip',     # 27
    # 'rsvp'       # 28
]

train_file_paths = ['Datasets/combined_train.csv']
test_file_paths = [
    'Datasets/CTU-13/final_dataset/test_1.csv',
    'Datasets/CTU-13/final_dataset/test_2.csv',
    'Datasets/CTU-13/final_dataset/test_5.csv',
    'Datasets/CTU-13/final_dataset/test_9.csv',
    'Datasets/CTU-13/final_dataset/test_13.csv',
    'Datasets/NCC/final_dataset/test_1.csv',
    'Datasets/NCC/final_dataset/test_2.csv',
    'Datasets/NCC/final_dataset/test_5.csv',
    'Datasets/NCC/final_dataset/test_9.csv',
    'Datasets/NCC/final_dataset/test_13.csv',
    'Datasets/NCC-2/final_dataset/test_1.csv',
    'Datasets/NCC-2/final_dataset/test_2.csv',
    'Datasets/NCC-2/final_dataset/test_3.csv',
]

log_file = "logs/final_classification_test_result_max_depth_29.log"

def log_message(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")


def stage1_label(label):
    label = str(label).strip().lower()
    return 0 if label == 'normal' else 1


def stage2_label(label):
    label = str(label).strip().lower()
    return 1 if 'spam' in label else 0

def multiclass_label(label):
    label = str(label).strip().lower()
    if label == 'normal':
        return 0
    elif 'spam' in label:
        return 2
    else:
        return 1
    
# Load and prepare training data
log_message(f'Processing dataset: {train_file_paths}')
df_train = pd.concat([pd.read_csv(p) for p in train_file_paths], ignore_index=True)
df_train['Stage1_Label'] = df_train['Label'].apply(stage1_label)
x_train_stage1 = df_train[selected_feature_cols]
y_train_stage1 = df_train['Stage1_Label']

# Train Stage 1 models ONCE
trained_stage1_models = {}
for stage1_name, stage1_model in models.items():
    log_message(f'--- Stage 1 Training with {stage1_name} ---')
    start_time = time.time()
    model = clone(stage1_model)
    model.fit(x_train_stage1, y_train_stage1)
    elapsed_time = time.time() - start_time
    log_message(f"Training Time: {elapsed_time:.5f} seconds")

    if 'Random Forest (Gini)' in stage1_name:
        depths = [tree.get_depth() for tree in model.estimators_]
        log_message(f"Random Forest Depths -> Avg: {np.mean(depths):.2f}, Max: {np.max(depths)}, Min: {np.min(depths)}")

    trained_stage1_models[stage1_name] = model

# Loop over test datasets
for test_path in test_file_paths:
    log_message(f'Processing dataset: {test_path}')
    df_test = pd.read_csv(test_path)
    df_test['Stage1_Label'] = df_test['Label'].apply(stage1_label)
    x_test_stage1 = df_test[selected_feature_cols]
    y_test_stage1 = df_test['Stage1_Label']

    for stage1_name, stage1_model in trained_stage1_models.items():
        log_message(f'--- Stage 1 Evaluation with {stage1_name} ---')

        start_time = time.time()
        y_pred_stage1 = stage1_model.predict(x_test_stage1)
        elapsed_time = time.time() - start_time
        log_message(f"Testing Time: {elapsed_time:.5f} seconds")

        acc = accuracy_score(y_test_stage1, y_pred_stage1)
        prec = precision_score(y_test_stage1, y_pred_stage1, average='weighted')
        rec = recall_score(y_test_stage1, y_pred_stage1, average='weighted')
        f1 = f1_score(y_test_stage1, y_pred_stage1, average='weighted')
        f2 = fbeta_score(y_test_stage1, y_pred_stage1, beta=2, average='weighted')

        log_message(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\nF2 Score: {f2:.4f}\n")
        log_message(classification_report(y_test_stage1, y_pred_stage1, digits=4))
        log_message(f"Confusion Matrix:\n{confusion_matrix(y_test_stage1, y_pred_stage1)}\n")

        # Stage 2: Train & Evaluate only if botnet samples exist
        train_stage2 = df_train[df_train['Stage1_Label'] == 1].copy()
        test_stage2 = df_test[y_pred_stage1 == 1].copy()

        if train_stage2.empty or test_stage2.empty:
            log_message("Skipping Stage 2 due to empty training or testing data.\n")
            continue

        train_stage2['Stage2_Label'] = train_stage2['Label'].apply(stage2_label)
        test_stage2['Stage2_Label'] = test_stage2['Label'].apply(stage2_label)

        x_train_stage2 = train_stage2[selected_feature_cols]
        y_train_stage2 = train_stage2['Stage2_Label']
        x_test_stage2 = test_stage2[selected_feature_cols]
        y_test_stage2 = test_stage2['Stage2_Label']

        for stage2_name, stage2_model in models.items():
            log_message(f'------ Stage 2 with {stage2_name} ------')

            model2 = clone(stage2_model)
            start_time = time.time()
            model2.fit(x_train_stage2, y_train_stage2)
            elapsed_time = time.time() - start_time
            log_message(f"Training Time: {elapsed_time:.5f} seconds")

            if 'Random Forest (Gini)' in stage2_name:
                depths = [tree.get_depth() for tree in model2.estimators_]
                log_message(f"Random Forest Depths (Stage 2) -> Avg: {np.mean(depths):.2f}, Max: {np.max(depths)}, Min: {np.min(depths)}")

            start_time = time.time()
            y_pred_stage2 = model2.predict(x_test_stage2)
            elapsed_time = time.time() - start_time
            log_message(f"Testing Time: {elapsed_time:.5f} seconds")

            acc2 = accuracy_score(y_test_stage2, y_pred_stage2)
            prec2 = precision_score(y_test_stage2, y_pred_stage2, average='weighted')
            rec2 = recall_score(y_test_stage2, y_pred_stage2, average='weighted')
            f1_2 = f1_score(y_test_stage2, y_pred_stage2, average='weighted')
            f2_2 = fbeta_score(y_test_stage2, y_pred_stage2, beta=2, average='weighted')

            log_message(f"Accuracy: {acc2:.4f}\nPrecision: {prec2:.4f}\nRecall: {rec2:.4f}\nF1 Score: {f1_2:.4f}\nF2 Score: {f2_2:.4f}\n")
            log_message(classification_report(y_test_stage2, y_pred_stage2, digits=4))
            log_message(f"Confusion Matrix:\n{confusion_matrix(y_test_stage2, y_pred_stage2)}\n")

            # Combined Evaluation
            combined_predictions = np.where(y_pred_stage1 == 0, 0, -1)
            botnet_indices = np.where(y_pred_stage1 == 1)[0]
            for i, idx in enumerate(botnet_indices):
                combined_predictions[idx] = y_pred_stage2[i] + 1

            y_true_combined = df_test['Label'].apply(multiclass_label)

            log_message("Combined Classification Report:")
            acc_c = accuracy_score(y_true_combined, combined_predictions)
            prec_c = precision_score(y_true_combined, combined_predictions, average='weighted')
            rec_c = recall_score(y_true_combined, combined_predictions, average='weighted')
            f1_c = f1_score(y_true_combined, combined_predictions, average='weighted')
            f2_c = fbeta_score(y_true_combined, combined_predictions, beta=2, average='weighted')

            log_message(f"Accuracy: {acc_c:.4f}\nPrecision: {prec_c:.4f}\nRecall: {rec_c:.4f}\nF1 Score: {f1_c:.4f}\nF2 Score: {f2_c:.4f}\n")
            log_message(classification_report(y_true_combined, combined_predictions, digits=4))
            log_message(f"Confusion Matrix:\n{confusion_matrix(y_true_combined, combined_predictions)}\n")

log_message('All loops completed.')