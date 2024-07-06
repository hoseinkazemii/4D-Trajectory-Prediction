import os
import pandas as pd


def _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params):
    verbose = params.get("verbose")
    report_directory = params.get("report_directory")
    if verbose:
        print(f"saving results...")

    if not os.path.exists("./Reports/"):
        os.makedirs("./Reports/")
    
    Y_true = Y_true.flatten()
    Y_pred = Y_pred.flatten()

    results_df = pd.DataFrame({
        "X_true": X_true,
        "Y_true": Y_true,
        "Z_true": Z_true,
        "X_predicted": X_pred,
        "Y_predicted": Y_pred,
        "Z_predicted": Z_pred
    })
    
    print("report_directory: ", report_directory) 
    results_path = os.path.join(report_directory, "Results.csv")
    results_df.to_csv(results_path, index=False)