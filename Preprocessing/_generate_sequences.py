import numpy as np

def _generate_sequences(timeseries, test_mode, **params):
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    sequence_step = params.get("sequence_step")
    test_stride_mode = params.get("test_stride_mode")

    total_window = sequence_length + prediction_horizon
    stride = prediction_horizon if test_stride_mode == "prediction_horizon" else total_window

    X, y = [], []

    if test_mode:
        for i in range(0, len(timeseries) - total_window + 1, stride):
            X_window = timeseries[i : i + sequence_length]
            y_window = timeseries[i + sequence_length : i + total_window, :3]

            X.append(X_window)
            y.append(y_window)
    else:
        for i in range(0, len(timeseries) - total_window + 1, sequence_step):
            X_window = timeseries[i : i + sequence_length]
            y_window = timeseries[i + sequence_length : i + total_window, :3]

            X.append(X_window)
            y.append(y_window)

    return np.array(X), np.array(y)
