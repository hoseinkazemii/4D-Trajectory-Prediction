import numpy as np

def _aggregate_sequence_predictions(y, row_counts, **params):
    """
    Aggregates predictions for multiple separate scenarios (timelines) without bridging.

    Parameters
    ----------
    y : np.ndarray
        Shape (total_sequences, prediction_horizon, num_features).
        This is the concatenated predictions (in scenario order).
    row_counts : list of int
        The row counts for each scenario in the dataset. We'll only consider
        the last 'num_test' entries for the test part (see 'num_test' below).
    params : dict
      - 'sequence_length' (int)
      - 'prediction_horizon' (int)
      - 'sequence_step' (int), often 1
      - 'num_test' (int): how many scenarios belong to the test set
      (any other needed params)

    Returns
    -------
    y_aggregated : np.ndarray
        A single array containing the concatenated aggregator results
        from each scenario. Each scenario i yields exactly
        (n_seq_i + prediction_horizon - 1) rows, where
           n_seq_i = (row_counts_i - sequence_length - prediction_horizon + 1) // sequence_step

    Explanation
    -----------
    1) We compute how many test sequences each scenario contributed, n_seq_i.
    2) We slice y into chunks [idx_start : idx_start + n_seq_i].
    3) For each chunk, we run _aggregate_single_scenario to produce (n_seq_i + horizon -1) rows.
    4) We do not let scenario i overlap with scenario i+1. The aggregator resets for each scenario.
    5) Finally, we concatenate scenario aggregator arrays => shape (sum of aggregator rows, num_features).
    """

    sequence_length     = params.get("sequence_length")
    prediction_horizon  = params.get("prediction_horizon")
    sequence_step       = params.get("sequence_step")
    test_indices = params.get("test_indices")


    test_row_counts = [row_counts[i] for i in test_indices]

    # Compute how many sequences each scenario contributed
    scenario_nseqs = []
    for c in test_row_counts:
        # n_seq_i = (rows_i - sequence_length - horizon + 1) // sequence_step
        n_seq = (c - sequence_length - prediction_horizon + 1) // sequence_step
        # If scenario doesn't have enough rows, n_seq might be <= 0
        scenario_nseqs.append(max(n_seq, 0))

    # Helper aggregator for a SINGLE scenario
    def _aggregate_single_scenario(y_chunk):
        """
        y_chunk : shape (n_seq, prediction_horizon, num_features)
        We'll produce exactly (n_seq + prediction_horizon - 1) aggregator rows.

        Approach:
        - final_len = n_seq + prediction_horizon - 1
        - sums, counts arrays of shape (final_len, num_features)
        - for each sequence i, for pred_idx in 0..(horizon-1):
            sums[i + pred_idx] += y_chunk[i, pred_idx]
            counts[i + pred_idx]++
        - aggregator = sums / counts
        """
        n_seq, _, n_feat = y_chunk.shape
        final_len = n_seq + prediction_horizon - 1

        sums = np.zeros((final_len, n_feat))
        counts = np.zeros(final_len, dtype=np.float32)

        for i in range(n_seq):
            for p in range(prediction_horizon):
                t = i + p  # local time index in [0..(final_len-1)]
                sums[t] += y_chunk[i, p]
                counts[t] += 1

        aggregator = np.zeros((final_len, n_feat), dtype=np.float32)
        for t in range(final_len):
            aggregator[t] = sums[t] / counts[t]
        return aggregator

    # Now slice the input y scenario-by-scenario
    aggregated_list = []
    idx_start = 0

    for i, n_seq in enumerate(scenario_nseqs):
        if n_seq <= 0:
            # This scenario doesn't produce any valid sequences
            continue

        idx_end = idx_start + n_seq
        # chunk shape => (n_seq, pred_horizon, num_features)
        y_chunk = y[idx_start: idx_end]
        idx_start = idx_end

        # aggregator for scenario i
        scen_agg = _aggregate_single_scenario(y_chunk)
        aggregated_list.append(scen_agg)

    # If no scenarios, return empty
    if len(aggregated_list) == 0:
        return np.array([])

    # Concatenate aggregator arrays from each scenario
    # shape => (sum_of_scenario_agg_rows, num_features)
    y_aggregated = np.concatenate(aggregated_list, axis=0)

    return y_aggregated


#############################
# import numpy as np

# def _aggregate_sequence_predictions(y, row_counts, **params):
#     """
#     Aggregates predictions for multiple scenarios without bridging across scenario boundaries.

#     y : np.ndarray, shape (total_sequences, prediction_horizon, num_features)
#         The combined test predictions for all scenarios, concatenated.

#     Returns:
#       y_aggregated : np.ndarray of shape (total_aggregated_points, num_features)
#         The scenario-by-scenario aggregator result, concatenated.

#     Explanation:
#       We figure out how many sequences each scenario contributed:
#         n_seq_i = (row_counts[i] - sequence_length - prediction_horizon + 1) // sequence_step
#       Then we aggregator each scenario chunk separately and concatenate.
#     """

#     sequence_length = params.get("sequence_length")
#     prediction_horizon = params.get("prediction_horizon")
#     sequence_step = params.get("sequence_step")
#     num_test = params.get("num_test")

#     # y shape => (total_sequences, prediction_horizon, num_features)
#     # We must figure out how many sequences belong to each scenario.
#     # For scenario i:
#     #   n_seq_i = (row_counts[i] - sequence_length - prediction_horizon + 1) // sequence_step
#     scenario_nseqs = []
#     # print(f"row_counts[-num_test:]: {row_counts[-num_test:]}")
#     for c in row_counts[-num_test:]:
#         # print(f"c: {c}")
#         n_seq = (c - sequence_length - prediction_horizon + 1) // sequence_step
#         scenario_nseqs.append(n_seq)
#     # print(f"scenario_nseqs: {scenario_nseqs}")

#     # We'll define a helper function that does the existing aggregator
#     # for a single scenario's chunk of shape (n_seq, prediction_horizon, num_features).
#     def _aggregate_single_scenario(y_chunk):
#         num_sequences, _, num_features = y_chunk.shape
#         point_sums = {}
#         point_counts = {}

#         for seq_idx in range(num_sequences):
#             for pred_idx in range(prediction_horizon):
#                 time_point = seq_idx + pred_idx
#                 if time_point not in point_sums:
#                     point_sums[time_point] = np.zeros(num_features)
#                     point_counts[time_point] = 0
#                 point_sums[time_point] += y_chunk[seq_idx, pred_idx]
#                 point_counts[time_point] += 1

#         total_points = max(point_sums.keys()) + 1
#         y_scen_agg = np.zeros((total_points, num_features))
#         for t in range(total_points):
#             y_scen_agg[t] = point_sums[t] / point_counts[t]

#         return y_scen_agg

#     # We'll slice the input y scenario by scenario and aggregator each chunk
#     aggregated_list = []
#     idx_start = 0
#     for i, n_seq in enumerate(scenario_nseqs):
#         if n_seq <= 0:
#             # This scenario might not produce sequences if row_counts[i] < sequence_length+prediction_horizon
#             continue

#         idx_end = idx_start + n_seq  # chunk for scenario i
#         y_chunk = y[idx_start:idx_end]   # shape => (n_seq, pred_horizon, num_features)
#         idx_start = idx_end

#         # aggregator for scenario i alone
#         scen_agg = _aggregate_single_scenario(y_chunk)
#         aggregated_list.append(scen_agg)

#     # Finally, we just concatenate the aggregator results from each scenario
#     if len(aggregated_list) == 0:
#         # No scenarios, return empty
#         return np.array([])

#     y_aggregated = np.concatenate(aggregated_list, axis=0)
#     return y_aggregated





###############################
# import numpy as np

# def _aggregate_sequence_predictions(y, **params):
#     prediction_horizon = params.get("prediction_horizon")
#     num_sequences, _, num_features = y.shape

#     # Initialize dictionaries to store sums and counts
#     point_sums = {}
#     point_counts = {}

#     # Fill the aggregator
#     for seq_idx in range(num_sequences):
#         for pred_idx in range(prediction_horizon):
#             time_point = seq_idx + pred_idx  # The time index in the "flattened" series

#             if time_point not in point_sums:
#                 point_sums[time_point] = np.zeros(num_features)
#                 point_counts[time_point] = 0
            
#             point_sums[time_point] += y[seq_idx, pred_idx]
#             point_counts[time_point] += 1

#     # Convert to final aggregated array
#     total_points = max(point_sums.keys()) + 1
#     y_aggregated = np.zeros((total_points, num_features))

#     # Compute average at each time point
#     for t in range(total_points):
#         y_aggregated[t] = point_sums[t] / point_counts[t]

#     return y_aggregated




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