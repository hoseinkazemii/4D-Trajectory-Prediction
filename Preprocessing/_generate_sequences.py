import numpy as np

def _generate_sequences(timeseries, **params):
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    sequence_step = params.get("sequence_step")

    total_window = sequence_length + prediction_horizon

    X, y = [], []

    # Slide over the timeseries with the given step
    # from t=0 up to the point where we can have a full sequence_length+prediction_horizon
    for i in range(0, len(timeseries) - total_window + 1, sequence_step):
        X_window = timeseries[i : i + sequence_length]     # shape (sequence_length, 9)
        y_window = timeseries[i + sequence_length : i + total_window, :3]  # shape (prediction_horizon, 3)




        X.append(X_window)
        y.append(y_window)

    X_array = np.array(X)
    y_array = np.array(y)

    return X_array, y_array
