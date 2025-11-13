import pandas as pd
import os

def compare_csv_pair(local_file, github_file, index_cols, comparison_name):
    print(f"\n{'='*20} Comparing: {comparison_name} {'='*20}")

    if not os.path.exists(local_file):
        print(f"Error: Local file not found: {local_file}")
        return None
    if not os.path.exists(github_file):
        print(f"Error: GitHub file not found: {github_file}")
        return None

    try:
        df_local = pd.read_csv(local_file)
        df_github = pd.read_csv(github_file)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return None

    if not all(col in df_local.columns for col in index_cols):
        print(f"Error: Local file {local_file} missing index columns: {index_cols}")
        return None
    if not all(col in df_github.columns for col in index_cols):
        print(f"Error: GitHub file {github_file} missing index columns: {index_cols}")
        return None

    try:
        df_local = df_local.set_index(index_cols)
        df_github = df_github.set_index(index_cols)
        
        df_local.sort_index(inplace=True)
        df_github.sort_index(inplace=True)
    except Exception as e:
        print(f"Error setting index: {e}")
        return None
        
    try:
        differences = df_local.compare(df_github, align_axis=0)
    except Exception as e:
        print(f"Error during comparison: {e}")
        print("This can happen if rows or columns do not align.")
        return None

    print("\n--- Inference ---")
    if differences.empty:
        print("Result: Files are identical.")
        print("Inference: Both files report the exact same results, likely from the same experiment.")
        return None
    else:
        print("Result: Files are different.")
        print("Inference: The files represent two different experimental runs. Metrics, times, and/or confusion matrices differ.")
        print("\n--- Differences Found (Local=self, GitHub=other) ---")
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        print(differences)
        
        differences.reset_index(inplace=True)
        differences['Comparison'] = comparison_name
        return differences

def main():
    all_differences = []
    
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
        diff_df = compare_csv_pair(
            local_file=pair['local'],
            github_file=pair['github'],
            index_cols=pair['index'],
            comparison_name=pair['name']
        )
        if diff_df is not None:
            all_differences.append(diff_df)
    
    print(f"\n{'='*25} Final Summary {'='*25}")
    if not all_differences:
        print("All file pairs are identical.")
        print("No consolidated difference CSV generated.")
    else:
        output_filename = 'all_comparison_differences.csv'
        consolidated_df = pd.concat(all_differences, ignore_index=True)
        
        cols = list(consolidated_df.columns)
        cols.remove('Comparison')
        cols.insert(0, 'Comparison')
        consolidated_df = consolidated_df[cols]
        
        consolidated_df.to_csv(output_filename, index=False)
        print(f"Successfully saved all consolidated differences to: {output_filename}")

if __name__ == "__main__":
    main()