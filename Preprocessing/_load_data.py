import pandas as pd
import os

def _load_data(**params):
    verbose = params.get("verbose", True)
    data_directory = params.get("data_directory")
    signal_based_extraction = params.get("signal_based_extraction")
    time_interval = params.get("time_interval")

    if verbose:
        print("Loading the datasets...")

    # Gather all CSV file paths
    all_files = sorted(
        [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.csv')]
    )

    # Read each CSV into its own DataFrame
    df_list = []
    for file_path in all_files:
        df = pd.read_csv(file_path)

        if signal_based_extraction:
            if verbose:
                print(f"Applying signal-based extraction to: {file_path}")
            
            df = df[df["LoadingStarted"] == 1]
            df = df[['Time', 'X', 'Y', 'Z']]

            # Sort by Time column just in case it's not sorted
            df = df.sort_values(by="Time").reset_index(drop=True)

            # Extract rows based on time intervals
            new_rows = []
            baseline_time = df.iloc[0]['Time']
            interval = time_interval

            while baseline_time <= df['Time'].iloc[-1]:
                next_row = df[df['Time'] >= baseline_time].iloc[0]  # First row with time >= baseline_time
                new_rows.append(next_row)
                baseline_time += interval

            # Create a new DataFrame from the extracted rows
            df = pd.DataFrame(new_rows)

        df_list.append(df)

    row_counts = [len(df) for df in df_list]

    # Instead of concatenating, we return df_list
    return df_list, row_counts


# import pandas as pd
# import os

# def _load_data(**params):
#     verbose = params.get("verbose", True)
#     data_directory = params.get("data_directory")

#     if verbose:
#         print("Loading the datasets...")

#     # Gather all CSV file paths
#     all_files = sorted(
#         [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.csv')]
#     )

#     # Read each CSV into its own DataFrame
#     df_list = []
#     for file_path in all_files:
#         df_list.append(pd.read_csv(file_path))

#     row_counts = [len(df) for df in df_list]
    
#     # Instead of concatenating, we return df_list
#     return df_list, row_counts
