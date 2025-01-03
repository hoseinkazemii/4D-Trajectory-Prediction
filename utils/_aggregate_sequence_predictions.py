import numpy as np


def _aggregate_sequence_predictions(y, **params):
    prediction_horizon = params.get("prediction_horizon")
    
    """
    Aggregate y for overlapping sequences by taking the mean for the overlapping points.
    
    Parameters:
    y (numpy.ndarray): Array of shape (num_sequences, prediction_horizon, 2) containing predicted coordinates.
    sequence_length (int): The length of the input sequence used for y.
    prediction_horizon (int): The length of the predicted sequence.
    
    Returns:
    numpy.ndarray: Aggregated y of shape (total_points, 2).
    """
    # Initialize a dictionary to store sums and counts of y for each point
    point_sums = {}
    point_counts = {}

    num_sequences = y.shape[0]

    for seq_idx in range(num_sequences):
        for pred_idx in range(prediction_horizon):
            time_point = seq_idx + pred_idx  # Calculate the time point in the overall series
            
            if time_point not in point_sums:
                point_sums[time_point] = np.zeros(2)
                point_counts[time_point] = 0
            
            point_sums[time_point] += y[seq_idx, pred_idx]
            point_counts[time_point] += 1

    # Calculate the mean for each point
    total_points = max(point_sums.keys()) + 1
    y_aggregated = np.zeros((total_points, 2))

    for time_point in range(total_points):
        y_aggregated[time_point] = point_sums[time_point] / point_counts[time_point]

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