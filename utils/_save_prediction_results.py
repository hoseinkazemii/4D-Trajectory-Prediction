import os
import pandas as pd


def _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, row_counts, **params):
    verbose = params.get("verbose")
    report_directory = params.get("report_directory")
    num_test = params.get("num_test")
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    sequence_step = params.get("sequence_step")
    if verbose:
        print(f"saving results...")

    if not os.path.exists("./Reports/"):
        os.makedirs("./Reports/")

    Y_true = Y_true.flatten()
    Y_pred = Y_pred.flatten()
    
    # Adjust row counts based on the preprocessing parameters
    adjusted_row_counts = [
        row_count - sequence_length - prediction_horizon + sequence_step
        for row_count in row_counts[-num_test:]
    ]

    # print(f"X_true.shape: {X_true.shape}")
    # print(f"adjusted_row_counts: {adjusted_row_counts}")
    # raise ValueError
    start_index = 0
    for i in range(num_test):
        end_index = start_index + adjusted_row_counts[i]

        # Create DataFrame for the current scenario
        results_df = pd.DataFrame({
            "X_true": X_true[start_index:end_index],
            "Y_true": Y_true[start_index:end_index],
            "Z_true": Z_true[start_index:end_index],
            "X_predicted": X_pred[start_index:end_index],
            "Y_predicted": Y_pred[start_index:end_index],
            "Z_predicted": Z_pred[start_index:end_index]
        })
        
        # Save to a separate CSV file
        scenario_filename = f"Results_TestSet_{i+1}.csv"
        results_path = os.path.join(report_directory, scenario_filename)
        results_df.to_csv(results_path, index=False)

        if verbose:
            print(f"Saved results for scenario {i+1} to {results_path}")

        # Update start index for the next scenario
        start_index = end_index


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