from ._generate_sequences import _generate_sequences

def _split_data_by_scenario(scaled_arrays_dict, row_counts, **params):
    """
    Dynamically splits each coordinate's scaled array into train/val/test sets,
    then generates sequences for each split.

    scaled_arrays_dict: dict
      e.g. {
        "Y":  <scaled np.ndarray, shape=(total_samples, 1)>,
        "XZ": <scaled np.ndarray, shape=(total_samples, 2)>,
        ...
      }
    row_counts: list of row counts per scenario file (used to sum up how many rows
                belong to train, val, test parts, etc.)
    params:
      - 'num_train', 'num_val', 'num_test': integers specifying how many scenario
        files go to train/val/test.
      - 'coordinates': list of coordinate strings
      - other arguments for _generate_sequences (e.g. 'sequence_length')

    Returns:
      split_data_dict: dict keyed by coordinate group, e.g. "Y", "XZ", "XYZ", ...
                       Each value is another dict containing:
                         {
                           "X_train": <array>,
                           "y_train": <array>,
                           "X_val":   <array>,
                           "y_val":   <array>,
                           "X_test":  <array>,
                           "y_test":  <array>
                         }
    """
    verbose = params.get("verbose", True)
    coordinates = params.get("coordinates")
    num_train_files = params.get("num_train")
    num_val_files = params.get("num_val")
    num_test_files = params.get("num_test")

    if verbose:
        print("Splitting the data by scenario into training, validation, and test sets...")
        print("Generating sequences for each coordinate group's timeseries data...")

    # Compute the total number of rows for each subset
    train_indices = sum(row_counts[:num_train_files])  # total rows for train
    val_indices = sum(row_counts[num_train_files : num_train_files + num_val_files])
    test_indices = sum(row_counts[num_train_files + num_val_files : 
                                  num_train_files + num_val_files + num_test_files])

    # This dictionary will hold all splits for each coordinate group
    split_data_dict = {}

    # Loop through each coordinate group
    for coord_str in coordinates:
        # Extract the scaled data array for this coordinate group
        data_scaled = scaled_arrays_dict[coord_str]  # shape: (total_rows, in_dim)

        # Partition the data into train, val, test
        train_data = data_scaled[:train_indices]
        val_data   = data_scaled[train_indices : train_indices + val_indices]
        test_data  = data_scaled[train_indices + val_indices : 
                                 train_indices + val_indices + test_indices]

        # Generate sequences
        # _generate_sequences typically returns (X, y) for each subset
        X_train, y_train = _generate_sequences(train_data, **params)
        X_val,   y_val   = _generate_sequences(val_data, **params)
        X_test,  y_test  = _generate_sequences(test_data, **params)

        # Store results in a sub-dictionary for this coordinate group
        split_data_dict[coord_str] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val":   X_val,
            "y_val":   y_val,
            "X_test":  X_test,
            "y_test":  y_test
        }

    return split_data_dict