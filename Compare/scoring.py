import pandas as pd
import os
import sys

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

def score_csv_pair(local_file, github_file, index_cols, metrics_to_score):
    print(f"\n{'='*20} Adversarial Scoring: {local_file} vs {github_file} {'='*20}")
    
    df_local = load_and_prep_df(local_file, index_cols)
    df_github = load_and_prep_df(github_file, index_cols)
    
    if df_local is None or df_github is None:
        print("Scoring skipped due to file errors.")
        return None
        
    try:
        if not df_local.index.equals(df_github.index):
            print("Warning: Indices do not match. Comparing intersecting rows only.", file=sys.stderr)
            common_index = df_local.index.intersection(df_github.index)
            df_local = df_local.loc[common_index]
            df_github = df_github.loc[common_index]
    except Exception as e:
        print(f"Error aligning DataFrames: {e}", file=sys.stderr)
        return None

    scores = {'Local_Wins': 0, 'GitHub_Wins': 0, 'Draws': 0}
    
    for metric, goal in metrics_to_score.items():
        if metric not in df_local.columns or metric not in df_github.columns:
            continue
            
        local_series = pd.to_numeric(df_local[metric], errors='coerce')
        github_series = pd.to_numeric(df_github[metric], errors='coerce')
        
        if goal == 'higher':
            scores['Local_Wins'] += (local_series > github_series).sum()
            scores['GitHub_Wins'] += (local_series < github_series).sum()
        elif goal == 'lower':
            scores['Local_Wins'] += (local_series < github_series).sum()
            scores['GitHub_Wins'] += (local_series > github_series).sum()
            
        scores['Draws'] += (local_series == github_series).sum()

    total_comparisons = scores['Local_Wins'] + scores['GitHub_Wins'] + scores['Draws']
    
    print("\n--- Terminal Scorecard ---")
    print(f"Total metric comparisons: {total_comparisons}")
    print(f"Local Wins (Local is better): {scores['Local_Wins']}")
    print(f"GitHub Wins (GitHub is better): {scores['GitHub_Wins']}")
    print(f"Draws (Values are identical): {scores['Draws']}")
    
    if scores['Local_Wins'] > scores['GitHub_Wins']:
        print("Inference: The 'local' file performed better overall in this comparison.")
    elif scores['GitHub_Wins'] > scores['Local_Wins']:
        print("Inference: The 'github' file performed better overall in this comparison.")
    else:
        print("Inference: The files performed equally in this comparison.")
        
    return {
        'Comparison': os.path.basename(local_file).replace('.csv', ''),
        'Local_Wins': scores['Local_Wins'],
        'GitHub_Wins': scores['GitHub_Wins'],
        'Draws': scores['Draws'],
        'Total_Comparisons': total_comparisons
    }

def main():
    pd.set_option('display.width', 1000)
    
    metric_cols = [
        'S1 Accuracy', 'S1 Precision', 'S1 Recall', 'S1 F1 Score',
        'S2 Accuracy', 'S2 Precision', 'S2 Recall', 'S2 F1 Score',
        'Combined Accuracy', 'Combined Precision', 'Combined Recall', 'Combined F1 Score'
    ]
    
    time_cols_classification = ['S1 Time (s)', 'S2 Time (s)']
    
    time_cols_feature = [
        'S1 Train Time (s)', 'S1 Test Time (s)', 
        'S2 Train Time (s)', 'S2 Test Time (s)'
    ]
    
    time_cols_final_test = [
        'S1 Train Time (s)', 'S1 Test Time (s)',
        'S2 Train Time (s)', 'S2 Test Time (s)'
    ]
    
    scoring_logic = {}
    for col in metric_cols:
        scoring_logic[col] = 'higher'
    for col in set(time_cols_classification + time_cols_feature + time_cols_final_test):
        scoring_logic[col] = 'lower'

    file_pairs = [
        {
            'local': 'classification_results.csv',
            'github': 'classification_resultsGithub.csv',
            'index': ['Stage 1 Model', 'Stage 2 Model'],
            'metrics': scoring_logic
        },
        {
            'local': 'final_test_results.csv',
            'github': 'final_test_resultsGithub.csv',
            'index': ['Test Dataset'],
            'metrics': scoring_logic
        },
        {
            'local': 'feature_n_results.csv',
            'github': 'feature_n_resultsGithub.csv',
            'index': ['Feature Count'],
            'metrics': scoring_logic
        }
    ]
    
    all_scores = []
    
    for pair in file_pairs:
        score_data = score_csv_pair(
            local_file=pair['local'],
            github_file=pair['github'],
            index_cols=pair['index'],
            metrics_to_score=pair['metrics']
        )
        if score_data:
            all_scores.append(score_data)
            
    print(f"\n{'='*25} Final Summary {'='*25}")
    if not all_scores:
        print("No comparisons were successfully completed.")
    else:
        output_filename = 'adversarial_score_summary.csv'
        summary_df = pd.DataFrame(all_scores)
        summary_df.to_csv(output_filename, index=False)
        print(f"Successfully saved all scores to: {output_filename}")

if __name__ == "__main__":
    main()