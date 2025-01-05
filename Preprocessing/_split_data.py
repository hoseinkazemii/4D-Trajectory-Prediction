import numpy as np

def _split_data(data, **params):
    verbose = params.get("verbose")
    sequence_length = params.get("sequence_length")
    train_data_split = params.get("train_data_split")
    if verbose:
        print("splitting the data...")

    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + 1:i + sequence_length + 1])
    
    X = np.array(X)
    y = np.array(y)

    # Manually split the data into training and testing sets
    split_index = int(len(X) * train_data_split)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test