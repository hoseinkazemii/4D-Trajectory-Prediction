import os
import pandas as pd
import numpy as np

def _save_prediction_results(X_true, Y_true, Z_true,
                             X_pred, Y_pred, Z_pred,
                             row_counts, **params):
    """
    Saves the aggregated predictions into multiple CSV files (one per test scenario),
    without duplicating rows at scenario boundaries.
    """
    verbose = params.get("verbose", True)
    report_directory = params.get("report_directory", "./Reports/")
    test_indices = params.get("test_indices")
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    sequence_step = params.get("sequence_step")

    if verbose:
        print(f"Saving results...")

    # Ensure the reports directory exists
    if not os.path.exists(report_directory):
        os.makedirs(report_directory)

    # Flatten Y arrays if you want to keep them 1D, as before
    # (X,Z might remain 1D if they are already 1D.)
    # But be consistent with your aggregator shape => if shape is (N, 3),
    # you might keep them as they are. We'll assume you want them 1D for Y only:
    if Y_true is not None:
        Y_true = Y_true.flatten()
    if Y_pred is not None:
        Y_pred = Y_pred.flatten()

    # Convert all to np arrays (in case they aren't)
    X_true = np.array(X_true) if X_true is not None else None
    Y_true = np.array(Y_true) if Y_true is not None else None
    Z_true = np.array(Z_true) if Z_true is not None else None
    X_pred = np.array(X_pred) if X_pred is not None else None
    Y_pred = np.array(Y_pred) if Y_pred is not None else None
    Z_pred = np.array(Z_pred) if Z_pred is not None else None

    # The aggregator output for each scenario is:
    # aggregator_length_i = n_seq_i + (prediction_horizon - 1)
    # where n_seq_i = (row_counts_i - sequence_length - prediction_horizon + 1) // sequence_step
    test_row_counts = [row_counts[i] for i in test_indices]
    aggregator_lengths = []
    for c in test_row_counts:
        n_seq = (c - sequence_length - prediction_horizon + 1) // sequence_step
        n_seq = max(n_seq, 0)
        agg_len_i = n_seq + (prediction_horizon - 1)
        aggregator_lengths.append(agg_len_i)

    start_index = 0
    for i, agg_len in enumerate(aggregator_lengths):
        if agg_len <= 0:
            # This scenario might not produce aggregator rows
            continue

        end_index = start_index + agg_len

        # Slice out the aggregator rows for scenario i
        X_true_slice = X_true[start_index:end_index] if X_true is not None else None
        Y_true_slice = Y_true[start_index:end_index] if Y_true is not None else None
        Z_true_slice = Z_true[start_index:end_index] if Z_true is not None else None
        X_pred_slice = X_pred[start_index:end_index] if X_pred is not None else None
        Y_pred_slice = Y_pred[start_index:end_index] if Y_pred is not None else None
        Z_pred_slice = Z_pred[start_index:end_index] if Z_pred is not None else None

        # Build DataFrame
        results_df = pd.DataFrame({
            "X_true":       X_true_slice if X_true_slice is not None else [],
            "Y_true":       Y_true_slice if Y_true_slice is not None else [],
            "Z_true":       Z_true_slice if Z_true_slice is not None else [],
            "X_predicted":  X_pred_slice if X_pred_slice is not None else [],
            "Y_predicted":  Y_pred_slice if Y_pred_slice is not None else [],
            "Z_predicted":  Z_pred_slice if Z_pred_slice is not None else []
        })

        scenario_filename = f"Results_TestSet_{i+1}.csv"
        results_path = os.path.join(report_directory, scenario_filename)
        results_df.to_csv(results_path, index=False)

        if verbose:
            print(f"Saved {agg_len} aggregator rows for scenario {i+1} to {results_path}")

        start_index = end_index




# import os
# import pandas as pd


# def _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, row_counts, **params):
#     verbose = params.get("verbose")
#     report_directory = params.get("report_directory")
#     num_test = params.get("num_test")
#     sequence_length = params.get("sequence_length")
#     prediction_horizon = params.get("prediction_horizon")
#     sequence_step = params.get("sequence_step")
#     if verbose:
#         print(f"saving results...")

#     if not os.path.exists("./Reports/"):
#         os.makedirs("./Reports/")

#     Y_true = Y_true.flatten()
#     Y_pred = Y_pred.flatten()
    
#     # Adjust row counts based on the preprocessing parameters
#     adjusted_row_counts = [
#         row_count - sequence_length - prediction_horizon + sequence_step
#         for row_count in row_counts[-num_test:]
#     ]
#     # print(f"adjusted_row_counts: {adjusted_row_counts}")
#     start_index = 0
#     for i in range(num_test):
#         end_index = start_index + adjusted_row_counts[i]

#         # Create DataFrame for the current scenario
#         results_df = pd.DataFrame({
#             "X_true": X_true[start_index:end_index],
#             "Y_true": Y_true[start_index:end_index],
#             "Z_true": Z_true[start_index:end_index],
#             "X_predicted": X_pred[start_index:end_index],
#             "Y_predicted": Y_pred[start_index:end_index],
#             "Z_predicted": Z_pred[start_index:end_index]
#         })
        
#         # Save to a separate CSV file
#         scenario_filename = f"Results_TestSet_{i+1}.csv"
#         results_path = os.path.join(report_directory, scenario_filename)
#         results_df.to_csv(results_path, index=False)

#         if verbose:
#             print(f"Saved results for scenario {i+1} to {results_path}")

#         # Update start index for the next scenario
#         start_index = end_index


# import os
# import pandas as pd


# def _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params):
#     verbose = params.get("verbose")
#     report_directory = params.get("report_directory")
#     if verbose:
#         print(f"saving results...")

#     if not os.path.exists("./Reports/"):
#         os.makedirs("./Reports/")
    
#     Y_true = Y_true.flatten()
#     Y_pred = Y_pred.flatten()

#     results_df = pd.DataFrame({
#         "X_true": X_true,
#         "Y_true": Y_true,
#         "Z_true": Z_true,
#         "X_predicted": X_pred,
#         "Y_predicted": Y_pred,
#         "Z_predicted": Z_pred
#     })
    
#     print("report_directory: ", report_directory) 
#     results_path = os.path.join(report_directory, "Results.csv")
#     results_df.to_csv(results_path, index=False)