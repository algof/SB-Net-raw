import pandas as pd
import os
import sys

def get_winner(local_val, github_val, goal):
    if pd.isna(local_val) or pd.isna(github_val):
        return 'No Data'
    
    local_val = pd.to_numeric(local_val, errors='coerce')
    github_val = pd.to_numeric(github_val, errors='coerce')
    
    if pd.isna(local_val) or pd.isna(github_val):
        return 'Conversion Error'

    if goal == 'higher':
        if local_val > github_val:
            return 'Local Win'
        elif local_val < github_val:
            return 'GitHub Win'
        else:
            return 'Draw'
    elif goal == 'lower':
        if local_val < github_val:
            return 'Local Win'
        elif local_val > github_val:
            return 'GitHub Win'
        else:
            return 'Draw'
    return 'N/A'

def load_and_prep_df(filepath, index_cols):
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return None
        
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None
        
    if not all(col in df.columns for col in index_cols):
        print(f"Error: {filepath} missing index columns: {index_cols}", file=sys.stderr)
        return None
        
    try:
        df = df.set_index(index_cols).sort_index()
    except Exception as e:
        print(f"Error setting index for {filepath}: {e}", file=sys.stderr)
        return None
        
    return df

def main():
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    metrics_to_score = {
        'S1 Accuracy': 'higher',
        'S1 Precision': 'higher',
        'S1 Recall': 'higher',
        'S1 F1 Score': 'higher',
        'S2 Accuracy': 'higher',
        'S2 Precision': 'higher',
        'S2 Recall': 'higher',
        'S2 F1 Score': 'higher',
        'Combined Accuracy': 'higher',
        'Combined Precision': 'higher',
        'Combined Recall': 'higher',
        'Combined F1 Score': 'higher',
        'Stage 1 Time (s)': 'lower',
        'Stage 2 Time (s)': 'lower',
        'S1 Train Time (s)': 'lower',
        'S1 Test Time (s)': 'lower',
        'S2 Train Time (s)': 'lower',
        'S2 Test Time (s)': 'lower'
    }

    file_pairs = [
        {
            'local': 'classification_results.csv',
            'github': 'classification_resultsGithub.csv',
            'index': ['Stage 1 Model', 'Stage 2 Model'],
            'name': 'Classification Results'
        },
        {
            'local': 'final_test_results.csv',
            'github': 'final_test_resultsGithub.csv',
            'index': ['Test Dataset'],
            'name': 'Final Test Results'
        },
        {
            'local': 'feature_n_results.csv',
            'github': 'feature_n_resultsGithub.csv',
            'index': ['Feature Count'],
            'name': 'Feature N Results'
        }
    ]

    for pair in file_pairs:
        print(f"\n{'='*20} Comparing: {pair['name']} {'='*20}")
        
        df_local = load_and_prep_df(pair['local'], pair['index'])
        df_github = load_and_prep_df(pair['github'], pair['index'])
        
        if df_local is None or df_github is None:
            print("Skipping comparison due to file error.")
            continue

        common_index = df_local.index.intersection(df_github.index)
        if len(common_index) < len(df_local.index) or len(common_index) < len(df_github.index):
             print(f"Warning: Indices do not match perfectly. Comparing {len(common_index)} common rows.")
        
        if len(common_index) == 0:
            print("Error: No common rows to compare. Skipping.")
            continue

        df_local = df_local.loc[common_index]
        df_github = df_github.loc[common_index]
        df_comparison = df_local.copy()

        terminal_summary = [f"\n--- Line-by-Line Comparison Summary for {pair['name']} ---"]
        
        for metric, goal in metrics_to_score.items():
            if metric not in df_local.columns or metric not in df_github.columns:
                continue

            winner_col = f"WINNER_{metric.replace(' ', '_')}"
            
            winners = [get_winner(l, g, goal) for l, g in zip(df_local[metric], df_github[metric])]
            df_comparison[winner_col] = winners
            
            counts = pd.Series(winners).value_counts()
            local_wins = counts.get('Local Win', 0)
            github_wins = counts.get('GitHub Win', 0)
            draws = counts.get('Draw', 0)
            
            summary_str = f"  {metric}: {local_wins} Local Wins, {github_wins} GitHub Wins, {draws} Draws"
            terminal_summary.append(summary_str)

        print("\n".join(terminal_summary))

        output_filename = f"detailed_comparison_{pair['local']}"
        df_comparison.reset_index().to_csv(output_filename, index=False)
        print(f"\nSaved detailed line-by-line comparison to: {output_filename}")

if __name__ == "__main__":
    main()