import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import os
import glob
import pathlib

def convert_port(port_val):
    if isinstance(port_val, str):
        try:
            return int(port_val)
        except ValueError:
            try:
                return int(port_val, 16)
            except (ValueError, TypeError):
                return -1
    elif pd.isna(port_val):
        return -1
    else:
        try:
            return int(port_val)
        except (ValueError, TypeError):
            return -1

def process_file_and_save(input_path, output_path):
    print(f"--- Processing: {input_path}")
    
    try:
        df = pd.read_csv(input_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.\n")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}\n")
        return
        
    binary_map = {
        True: 1, False: 0,
        'True': 1, 'False': 0,
        'true': 1, 'false': 0,
        1: 1, 0: 0,
        '1': 1, '0': 0,
        '1.0': 1, '0.0': 0
    }

    binary_cols = [
        'arp', 'esp', 'icmp', 'igmp', 'ipv6', 'ipv6-icmp', 
        'ipx/spx', 'llc', 'pim', 'rarp', 'rtcp', 'rtp', 'tcp', 'udp', 'udt', 'unas'
    ]
    
    categorical_cols = ['SrcAddr', 'Sport', 'Dir', 'DstAddr', 'Dport']
    
    new_cols_to_add = ['gre', 'rsvp']
    
    label_col = 'Label'
    
    output_column_order = [
        'Dur', 'SrcAddr', 'Sport', 'Dir', 'DstAddr', 'Dport', 'State', 
        'TotPkts', 'TotBytes', 'SrcBytes', 'Label', 'arp', 'esp', 'gre', 
        'icmp', 'igmp', 'ipv6', 'ipv6-icmp', 'ipx/spx', 'llc', 'pim', 
        'rarp', 'rsvp', 'rtcp', 'rtp', 'tcp', 'udp', 'udt', 'unas'
    ]

    if label_col in df.columns:
        y = df[[label_col]].astype(str)
        x = df.drop(columns=[label_col])
    else:
        y = None
        x = df.copy()

    for col in binary_cols:
        if col in x.columns:
            x[col] = x[col].map(binary_map).fillna(0).astype(np.int64)

    if 'Sport' in x.columns:
        x['Sport'] = x['Sport'].apply(convert_port)
    if 'Dport' in x.columns:
        x['Dport'] = x['Dport'].apply(convert_port)

    for col in new_cols_to_add:
        if col not in x.columns:
            x[col] = 0

    cats_to_process = [col for col in categorical_cols if col in x.columns]
    if cats_to_process:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        x[cats_to_process] = x[cats_to_process].astype(str).fillna("UNKNOWN")
        x[cats_to_process] = encoder.fit_transform(x[cats_to_process]).astype(np.int64)

    
    remaining_cols = set(x.columns) - set(binary_cols) - set(cats_to_process) - set(new_cols_to_add)
    for col in remaining_cols:
        x[col] = pd.to_numeric(x[col], errors='coerce').fillna(0)
        
        if (x[col] % 1 == 0).all():
            x[col] = x[col].astype(np.int64)
        else:
            x[col] = x[col].astype(np.float64)

    if y is not None:
        df_processed = pd.concat([x, y], axis=1)
    else:
        df_processed = x

    final_cols = [col for col in output_column_order if col in df_processed.columns]
    df_final = df_processed[final_cols]
    
    try:
        output_dir = pathlib.Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_final.to_csv(output_path, index=False)
        print(f"Successfully saved to '{output_path}'.\n")
    except Exception as e:
        print(f"Error: Could not save file '{output_path}'. {e}\n")


def main():
    dir_mappings = {
        'CTU-13/final_dataset/': 'CTU-13/processed/',
        'NCC/final_dataset/': 'NCC/processed/',
        'NCC-2/final_dataset/': 'NCC-2/processed/'
    }
    
    print("Starting batch preprocessing...")
    total_files_processed = 0
    
    for input_dir, output_dir in dir_mappings.items():
        print(f"Scanning directory: {input_dir}")
        search_pattern = os.path.join(input_dir, '*.csv')
        csv_files = glob.glob(search_pattern)
        
        if not csv_files:
            print(f"No CSV files found in {input_dir}")
            continue
            
        print(f"Found {len(csv_files)} files.")
        
        for input_file_path in csv_files:
            file_name = os.path.basename(input_file_path)
            output_file_path = os.path.join(output_dir, file_name)
            
            process_file_and_save(input_file_path, output_file_path)
            total_files_processed += 1
            
    print(f"Batch processing complete. Total files processed: {total_files_processed}")

if __name__ == "__main__":
    main()