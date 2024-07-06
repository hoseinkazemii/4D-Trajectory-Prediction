# Overlapping sequences generation
import numpy as np

def _generate_sequences(timeseries, **params):
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")

    X, y = [], []
    for i in range(len(timeseries) - sequence_length - prediction_horizon + 1):
        X.append(timeseries[i:i + sequence_length])
        y.append(timeseries[i + sequence_length:i + sequence_length + prediction_horizon])

    X_array = np.array(X)
    y_array = np.array(y)

    return X_array, y_array


# Non-overlapping sequences generation
# import numpy as np

# def _generate_sequences(timeseries, **params):
#     sequence_length = params.get("sequence_length")
#     prediction_horizon = params.get("prediction_horizon")  # Number of steps ahead to predict

#     X, y = [], []
#     for i in range(0, len(timeseries) - sequence_length - prediction_horizon + 1, sequence_length):
#         print(f"from {i} until {i + sequence_length} for X")
#         print(f"from {i + sequence_length} until {i + sequence_length + prediction_horizon} for y")
#         print("*****************************************")
#         X.append(timeseries[i:i + sequence_length])
#         y.append(timeseries[i + sequence_length:i + sequence_length + prediction_horizon])

#     X_array = np.array(X)
#     y_array = np.array(y)


#     return X_array, y_array