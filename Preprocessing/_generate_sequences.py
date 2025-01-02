# Overlapping sequences generation
import numpy as np

def _generate_sequences(timeseries, **params):
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    # sequence_step = params.get("sequence_step")

    sequence_step = sequence_length + prediction_horizon
    X, y = [], []
    # Step by the full window size to avoid overlap
    for i in range(0, len(timeseries) - sequence_length - prediction_horizon + 1, 1):
        X.append(timeseries[i:i + sequence_length])
        y.append(timeseries[i + sequence_length:i + sequence_length + prediction_horizon])

    X_array = np.array(X)
    y_array = np.array(y)

    return X_array, y_array




def generate_sequences_rolling_origin(timeseries, min_train_size, step_size=1, **params):
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    # sequence_step = params.get("sequence_step")

    sequences = []
    
    # Generate multiple train/test splits
    for i in range(min_train_size, len(timeseries) - prediction_horizon + 1, step_size):
        # Split point moves forward
        train = timeseries[:i]
        test = timeseries[i:i + prediction_horizon]
        
        # Use last sequence_length points for input
        X = train[-sequence_length:]
        y = test
        
        sequences.append((X, y))
    
    return sequences

# Example usage:
# If timeseries = [1,2,3,4,5,6,7,8,9,10]
# sequence_length = 3
# prediction_horizon = 2
# min_train_size = 5
# 
# You'll get splits like:
# Train on [1,2,3,4,5] to predict [6,7]
# Train on [1,2,3,4,5,6] to predict [7,8]
# Train on [1,2,3,4,5,6,7] to predict [8,9]