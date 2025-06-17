import os
import pandas as pd
import numpy as np

def _save_prediction_results(X_true, Y_true, Z_true,
                             X_pred, Y_pred, Z_pred,
                             row_counts, test_mode, **params):
    verbose = params.get("verbose", True)
    report_directory = params.get("report_directory")
    test_indices = params.get("test_indices")
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    sequence_step = params.get("sequence_step")
    test_stride_mode = params.get("test_stride_mode")

    if verbose:
        print(f"Saving results... test_mode={test_mode}")

    if not os.path.exists(report_directory):
        os.makedirs(report_directory)

    X_true = np.array(X_true) if X_true is not None else None
    Y_true = np.array(Y_true) if Y_true is not None else None
    Z_true = np.array(Z_true) if Z_true is not None else None
    X_pred = np.array(X_pred) if X_pred is not None else None
    Y_pred = np.array(Y_pred) if Y_pred is not None else None
    Z_pred = np.array(Z_pred) if Z_pred is not None else None

    total_window = sequence_length + prediction_horizon
    stride = prediction_horizon if test_stride_mode == "prediction_horizon" else total_window
    test_row_counts = [row_counts[i] for i in test_indices]

    aggregator_lengths = []
    for c in test_row_counts:
        diff = c - total_window
        if test_mode:
            n_seq = (diff // stride) + 1 if diff >= 0 else 0
            agg_len_i = n_seq * prediction_horizon
        else:
            n_seq = (diff // sequence_step) + 1 if diff >= 0 else 0
            agg_len_i = n_seq + (prediction_horizon - 1) if n_seq > 0 else 0
        aggregator_lengths.append(agg_len_i)

    start_index = 0
    for scenario_idx, agg_len in enumerate(aggregator_lengths, start=1):
        if agg_len <= 0:
            print(f"Scenario {scenario_idx} => 0 aggregator rows, skipping.")
            continue

        end_index = start_index + agg_len
        results_df = pd.DataFrame({
            "X_true": X_true[start_index:end_index] if X_true is not None else [],
            "Y_true": Y_true[start_index:end_index] if Y_true is not None else [],
            "Z_true": Z_true[start_index:end_index] if Z_true is not None else [],
            "X_predicted": X_pred[start_index:end_index] if X_pred is not None else [],
            "Y_predicted": Y_pred[start_index:end_index] if Y_pred is not None else [],
            "Z_predicted": Z_pred[start_index:end_index] if Z_pred is not None else []
        })

        scenario_filename = f"Results_TestSet_{scenario_idx}.csv"
        results_path = os.path.join(report_directory, scenario_filename)
        results_df.to_csv(results_path, index=False)

        if verbose:
            print(f"Saved {agg_len} aggregator rows for scenario {scenario_idx} to {results_path}")

        start_index = end_index
