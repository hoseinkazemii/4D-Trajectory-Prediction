# utils/log_results.py
import os
import pandas as pd
import numpy as np

def aggregate_predictions(y_true, y_pred, sequence_length):
    num_sequences = y_true.shape[0]
    feature_dim = y_true.shape[2]
    total_length = num_sequences + sequence_length - 1
    true_aggregated = np.zeros((total_length, feature_dim))
    pred_aggregated = np.zeros((total_length, feature_dim))
    counts = np.zeros(total_length)

    for i in range(num_sequences):
        for j in range(sequence_length):
            true_aggregated[i + j] += y_true[i, j]
            pred_aggregated[i + j] += y_pred[i, j]
            counts[i + j] += 1

    true_aggregated /= counts[:, None]
    pred_aggregated /= counts[:, None]

    return true_aggregated, pred_aggregated

def log_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, results_folder="./Results/", verbose=True):
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    results_df = pd.DataFrame({
        "X_true": X_true,
        "Y_true": Y_true,
        "Z_true": Z_true,
        "X_predicted": X_pred,
        "Y_predicted": Y_pred,
        "Z_predicted": Z_pred
    })
    
    results_path = os.path.join(results_folder, "results.csv")
    results_df.to_csv(results_path, index=False)
    
    if verbose:
        print(f"Results saved to {results_path}")