import pandas as pd
import os

def _load_data(**params):
    verbose = params.get("verbose", True)
    data_directory = params.get("data_directory")

    if verbose:
        print("Loading the datasets...")

    # Gather all CSV file paths
    all_files = sorted(
        [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.csv')]
    )

    # Read each CSV into its own DataFrame
    df_list = []
    for file_path in all_files:
        df_list.append(pd.read_csv(file_path))

    row_counts = [len(df) for df in df_list]
    
    # Instead of concatenating, we return df_list
    return df_list, row_counts




# import pandas as pd
# import os

# def _load_data(**params):
#     verbose = params.get("verbose")
#     data_directory = params.get("data_directory")
#     if verbose:
#         print("Loading the datasets...")

#     # Load all CSV files in the directory
#     all_files = [os.path.join(data_directory, file) for file in os.listdir(data_directory) if file.endswith('.csv')]

#     # Read each CSV file into a DataFrame and record the row counts
#     df_list = [pd.read_csv(file) for file in all_files]
#     row_counts = [len(df) for df in df_list]

#     # Concatenate all DataFrames into one
#     combined_df = pd.concat(df_list, ignore_index=True)

#     return combined_df, row_counts