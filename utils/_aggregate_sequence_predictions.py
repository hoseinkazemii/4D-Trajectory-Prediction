import numpy as np

def _aggregate_sequence_predictions(y, **params):
    prediction_horizon = params.get("prediction_horizon")
    num_sequences, _, num_features = y.shape

    # Initialize dictionaries to store sums and counts
    point_sums = {}
    point_counts = {}

    # Fill the aggregator
    for seq_idx in range(num_sequences):
        for pred_idx in range(prediction_horizon):
            time_point = seq_idx + pred_idx  # The time index in the "flattened" series

            if time_point not in point_sums:
                point_sums[time_point] = np.zeros(num_features)
                point_counts[time_point] = 0
            
            point_sums[time_point] += y[seq_idx, pred_idx]
            point_counts[time_point] += 1

    # Convert to final aggregated array
    total_points = max(point_sums.keys()) + 1
    y_aggregated = np.zeros((total_points, num_features))

    # Compute average at each time point
    for t in range(total_points):
        y_aggregated[t] = point_sums[t] / point_counts[t]

    return y_aggregated




if __name__ == "__main__":

    # Example usage
    y_test_XZ_coordinate = np.array([
        [[-1, 1], [-2, 2], [-3, 3], [-4, 4], [-5, 5], [-6, 6], [-7, 7]],
        [[-2, 2], [-3, 3], [-4, 4], [-5, 5], [-6, 6], [-7, 7], [-8, 8]],
        [[-3, 3], [-4, 4], [-5, 5], [-6, 6], [-7, 7], [-8, 8], [-9, 9]],
        [[-4, 4], [-5, 5], [-6, 6], [-7, 7], [-8, 8], [-9, 9], [-10, 10]]
    ])

    # Same as y_test_XZ_coordinate for easier checking
    y_pred_XZ_coordinate = np.array([
        [[-1, 1], [-2, 2], [-3, 3], [-4, 4], [-5, 5], [-6, 6], [-7, 7]],
        [[-2, 2], [-3, 3], [-4, 4], [-5, 5], [-6, 6], [-7, 7], [-8, 8]],
        [[-3, 3], [-4, 4], [-5, 5], [-6, 6], [-7, 7], [-8, 8], [-9, 9]],
        [[-4, 4], [-5, 5], [-6, 6], [-7, 7], [-8, 8], [-9, 9], [-10, 10]]
    ])

    prediction_horizon = 7  # Example value

    aggregated_predictions = _aggregate_sequence_predictions(y_test_XZ_coordinate, **{"prediction_horizon":7})
    print("Aggregated Predictions:\n", aggregated_predictions)