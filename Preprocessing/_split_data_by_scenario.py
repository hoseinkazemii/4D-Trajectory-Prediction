import numpy as np

from ._generate_sequences import _generate_sequences

def _split_data_by_scenario(Y_data_scaled, XZ_data_scaled, row_counts, **params):   
    verbose = params.get("verbose")
    coordinates = params.get("coordinates")
    # Determine the indices for training, validation, and test sets
    num_train_files = params.get("num_train")
    num_val_files = params.get("num_val")
    num_test_files = params.get("num_test")
    if verbose:
        print("Splitting the data by scenario into training, validation, and test sets...")
        print("Generating seuqences for coordinates timeseries data...")


    train_indices = sum(row_counts[:num_train_files])
    val_indices = sum(row_counts[num_train_files:num_train_files + num_val_files])
    test_indices = sum(row_counts[num_train_files + num_val_files:num_train_files + num_val_files + num_test_files])

    for coordinate in coordinates:
        if coordinate == "Y":
            Y_train_data = Y_data_scaled[:train_indices]
            Y_val_data = Y_data_scaled[train_indices:train_indices + val_indices]
            Y_test_data = Y_data_scaled[train_indices + val_indices:train_indices + val_indices + test_indices]
            X_train_Y_coordinate, y_train_Y_coordinate = _generate_sequences(Y_train_data, **params)
            X_val_Y_coordinate, y_val_Y_coordinate = _generate_sequences(Y_val_data, **params)
            X_test_Y_coordinate, y_test_Y_coordinate = _generate_sequences(Y_test_data, **params)
        if coordinate == "XZ":
            XZ_train_data = XZ_data_scaled[:train_indices]
            XZ_val_data = XZ_data_scaled[train_indices:train_indices + val_indices]
            XZ_test_data = XZ_data_scaled[train_indices + val_indices:train_indices + val_indices + test_indices]
            X_train_XZ_coordinate, y_train_XZ_coordinate = _generate_sequences(XZ_train_data, **params)
            X_val_XZ_coordinate, y_val_XZ_coordinate = _generate_sequences(XZ_val_data, **params)
            X_test_XZ_coordinate, y_test_XZ_coordinate = _generate_sequences(XZ_test_data, **params)      


    return X_train_Y_coordinate, X_val_Y_coordinate, X_test_Y_coordinate, y_train_Y_coordinate, y_val_Y_coordinate, y_test_Y_coordinate, \
        X_train_XZ_coordinate, X_val_XZ_coordinate, X_test_XZ_coordinate, y_train_XZ_coordinate, y_val_XZ_coordinate, y_test_XZ_coordinate