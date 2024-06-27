def _sequential_split(X_train, y_train, **params):
    validation_split = params.get("validation_split")

    train_size = int(len(X_train) * (1 - validation_split))
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]

    return X_train, X_val, y_train, y_val
