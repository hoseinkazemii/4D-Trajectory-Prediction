import numpy as np

def _split_data_by_scenario(scaled_data, row_counts, combined, **params):
    verbose = params.get("verbose")
    sequence_length = params.get("sequence_length")

    if verbose:
        print("splitting the data by scenario...")

    # Determine the indices for training, validation, and test sets
    num_train_files = params.get("num_train")
    num_val_files = params.get("num_val")
    num_test_files = params.get("num_test")

    train_indices = sum(row_counts[:num_train_files])
    val_indices = sum(row_counts[num_train_files:num_train_files + num_val_files])
    test_indices = sum(row_counts[num_train_files + num_val_files:num_train_files + num_val_files + num_test_files])

    train_data = scaled_data[:train_indices]
    val_data = scaled_data[train_indices:train_indices + val_indices]
    test_data = scaled_data[train_indices + val_indices:train_indices + val_indices + test_indices]

    def process_data(data):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + 1:i + sequence_length + 1])
        return np.array(X), np.array(y)

    X_train, y_train = process_data(train_data)
    X_val, y_val = process_data(val_data)
    X_test, y_test = process_data(test_data)

    return X_train, X_val, X_test, y_train, y_val, y_test