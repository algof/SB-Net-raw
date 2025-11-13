import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import time


# Define models
models = {
    'Random Forest (Gini)': RandomForestClassifier(criterion='gini', max_depth=29, random_state=42),
}

feature_cols = [
    # 'SrcAddr',   # 0
    # 'Sport',     # 1
    # 'Dport',     # 2
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
    'rtcp',      # 15
    'ipx/spx',   # 16
    'ipv6-icmp', # 17
    'arp',       # 18
    'pim',       # 19
    'rarp',      # 20
    'esp',       # 21
    'ipv6',      # 22
    'llc',       # 23
    'gre',       # 24
    'unas',      # 25
    'udt',       # 26
    'ipnip',     # 27
    'rsvp'       # 28
]


train_file_paths = ['Balancing/train-smote.csv']
test_file_paths = ['Datasets/train-test.csv']

log_file = "logs/RF-RF_maxDepth_29_results_SMOTE.log"

def log_message(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

def stage1_label(label):
    return 0 if label == 'normal' else 1

def stage2_label(label):
    return 1 if 'spam' in label else 0

for train_path, test_path in zip(train_file_paths, test_file_paths):
    log_message(f'Processing dataset: {train_path}')
    df_train = pd.read_csv(train_path)
    df_train = df_train.drop(
        columns=feature_cols, 
        errors='ignore'
    )
    df_train['Stage1_Label'] = df_train['Label'].apply(stage1_label)
    x_train_stage1 = df_train.drop(columns=['Label', 'Stage1_Label'])
    y_train_stage1 = df_train['Stage1_Label']
    
    #!################################################################
    #! Stage 1
    #!################################################################
    for stage1_name, stage1_model in models.items():

        log_message(f'--- Stage 1 with {stage1_name} ---')

        start_time = time.time()
        stage1_model.fit(x_train_stage1, y_train_stage1)
        end_time = time.time()
        elapsed_time = end_time - start_time
        log_message(f"Training Time: {elapsed_time:.5f} seconds")

        # todo: Log tree depth for Decision Tree or Random Forest
        if 'Decision Tree (Gini)' in stage1_name:
            log_message(f"Tree Depth curr: {stage1_model.get_depth()}")
        elif 'Random Forest (Gini)' in stage1_name:
            tree_depths = [tree.get_depth() for tree in stage1_model.estimators_]
            avg_depth = np.mean(tree_depths)
            max_depth = np.max(tree_depths)
            min_depth = np.min(tree_depths)
            log_message(f"Random Forest Depths curr -> Avg: {avg_depth:.2f}, Max: {max_depth}, Min: {min_depth}")
        # todo: Log tree depth for Decision Tree or Random Forest

        df_test = pd.read_csv(test_path)
        df_test = df_test.drop(
            columns=feature_cols, 
            errors='ignore'
        )
        df_test['Stage1_Label'] = df_test['Label'].apply(stage1_label)
        x_test_stage1 = df_test.drop(columns=['Label', 'Stage1_Label'])
        y_test_stage1 = df_test['Stage1_Label']

        start_time = time.time()
        y_pred_stage1 = stage1_model.predict(x_test_stage1)
        end_time = time.time()
        elapsed_time = end_time - start_time
        log_message(f"Testing Time: {elapsed_time:.5f} seconds")

        # Stage 1 evaluation
        accuracy_bc = accuracy_score(y_test_stage1, y_pred_stage1)
        precision_bc = precision_score(y_test_stage1, y_pred_stage1, average='macro')
        recall_bc = recall_score(y_test_stage1, y_pred_stage1, average='macro')
        f1_bc = f1_score(y_test_stage1, y_pred_stage1, average='macro')
        f2_bc = fbeta_score(y_test_stage1, y_pred_stage1, beta=2, average='macro')

        log_message(f"Accuracy: {accuracy_bc:.4f}\nPrecision: {precision_bc:.4f}\nRecall: {recall_bc:.4f}\nF1 Score: {f1_bc:.4f}\nF2 Score: {f2_bc:.4f}\n")
        log_message(classification_report(y_test_stage1, y_pred_stage1, digits=4))
        log_message(f"Confusion Matrix:\n{confusion_matrix(y_test_stage1, y_pred_stage1)}\n")

        # Stage 2 - Filter for botnet traffic only
        train_stage2 = df_train[df_train['Stage1_Label'] == 1].copy()
        test_stage2 = df_test[y_pred_stage1 == 1].copy()
        train_stage2['Stage2_Label'] = train_stage2['Label'].apply(stage2_label)
        test_stage2['Stage2_Label'] = test_stage2['Label'].apply(stage2_label)
        
        x_train_stage2 = train_stage2.drop(columns=['Label', 'Stage1_Label', 'Stage2_Label'])
        y_train_stage2 = train_stage2['Stage2_Label']
        x_test_stage2 = test_stage2.drop(columns=['Label', 'Stage1_Label', 'Stage2_Label'])
        y_test_stage2 = test_stage2['Stage2_Label']
        
        #?################################################################
        #? Stage 2
        #?################################################################
        for stage2_name, stage2_model in models.items():
            log_message(f'------ Stage 2 with {stage2_name} ------')

            start_time = time.time()
            stage2_model.fit(x_train_stage2, y_train_stage2)
            end_time = time.time()
            elapsed_time = end_time - start_time
            log_message(f"Training Time: {elapsed_time:.5f} seconds")

            if stage2_name == 'Decision Tree (Gini)':
                log_message(f"Tree Depth (Stage 2): {stage2_model.get_depth()}")
            elif stage2_name == 'Random Forest (Gini)':
                tree_depths = [tree.get_depth() for tree in stage2_model.estimators_]
                avg_depth = np.mean(tree_depths)
                max_depth = np.max(tree_depths)
                min_depth = np.min(tree_depths)
                log_message(f"Random Forest Depths (Stage 2) -> Avg: {avg_depth:.2f}, Max: {max_depth}, Min: {min_depth}")

            start_time = time.time()
            y_pred_stage2 = stage2_model.predict(x_test_stage2)
            end_time = time.time()
            elapsed_time = end_time - start_time
            log_message(f"Testing Time: {elapsed_time:.5f} seconds")

            # Stage 2 evaluation
            accuracy_stage2 = accuracy_score(y_test_stage2, y_pred_stage2)
            precision_stage2 = precision_score(y_test_stage2, y_pred_stage2, average='weighted')
            recall_stage2 = recall_score(y_test_stage2, y_pred_stage2, average='weighted')
            f1_stage2 = f1_score(y_test_stage2, y_pred_stage2, average='weighted')
            f2_stage2 = fbeta_score(y_test_stage2, y_pred_stage2, beta=2, average='weighted')
            
            log_message(f"Accuracy: {accuracy_stage2:.4f}\nPrecision: {precision_stage2:.4f}\nRecall: {recall_stage2:.4f}\nF1 Score: {f1_stage2:.4f}\nF2 Score: {f2_stage2:.4f}\n")
            log_message(classification_report(y_test_stage2, y_pred_stage2, digits=4))
            log_message(f"Confusion Matrix:\n{confusion_matrix(y_test_stage2, y_pred_stage2)}\n")

            # Combined evaluation
            combined_predictions = np.where(y_pred_stage1 == 0, 0, -1)
            botnet_indices = np.where(y_pred_stage1 == 1)[0]
            for i, idx in enumerate(botnet_indices):
                combined_predictions[idx] = y_pred_stage2[i] + 1

            y_true_combined = df_test['Label'].apply(lambda x: 0 if x == 'normal' else (1 if 'spam' not in x else 2))

            #~################################################################
            #~ Combined
            #~################################################################
            log_message("Combined Classification Report:")
            # combined evaluation
            accuracy_combined = accuracy_score(y_true_combined, combined_predictions)
            precision_combined = precision_score(y_true_combined, combined_predictions, average='weighted')
            recall_combined = recall_score(y_true_combined, combined_predictions, average='weighted')
            f1_combined = f1_score(y_true_combined, combined_predictions, average='weighted')
            f2_combined = fbeta_score(y_true_combined, combined_predictions, beta=2, average='weighted')
            log_message(f"Accuracy: {accuracy_combined:.4f}\nPrecision: {precision_combined:.4f}\nRecall: {recall_combined:.4f}\nF1 Score: {f1_combined:.4f}\nF2 Score: {f2_combined:.4f}\n")

            log_message(classification_report(y_true_combined, combined_predictions, digits=4))
            log_message(f"Confusion Matrix:\n{confusion_matrix(y_true_combined, combined_predictions)}\n")


log_message('All loops completed.')
