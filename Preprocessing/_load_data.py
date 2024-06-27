import pandas as pd
import os

def _load_data(**params):
    verbose = params.get("verbose")
    data_directory = params.get("data_directory")
    if verbose:
        print("loading the data...")

    # Load all CSV files in the directory
    all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.csv')]
    
    # Read each CSV file into a DataFrame and record the row counts
    df_list = [pd.read_csv(file) for file in all_files]
    row_counts = [len(df) for df in df_list]
    
    # Concatenate all DataFrames into one
    combined_df = pd.concat(df_list, ignore_index=True)

    
    return combined_df, row_counts