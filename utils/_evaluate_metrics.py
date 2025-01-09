import json
import csv
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params):
    """
    Computes a variety of metrics for 3D regression tasks.
    Returns a dictionary of metric_name -> value.
    """
    verbose = params.get("verbose")

    if verbose:
        print("Computing metrics...")
    metrics_dict = {}

    # Convert to numpy arrays (if they aren't already)
    X_true = np.array(X_true)
    Y_true = np.array(Y_true)
    Z_true = np.array(Z_true)
    X_pred = np.array(X_pred)
    Y_pred = np.array(Y_pred)
    Z_pred = np.array(Z_pred)

    # -----------------------------
    # 1. Coordinate-wise metrics
    # -----------------------------
    # MSE for each coordinate
    mse_x = mean_squared_error(X_true, X_pred)
    mse_y = mean_squared_error(Y_true, Y_pred)
    mse_z = mean_squared_error(Z_true, Z_pred)

    # RMSE for each coordinate
    rmse_x = np.sqrt(mse_x)
    rmse_y = np.sqrt(mse_y)
    rmse_z = np.sqrt(mse_z)

    # MAE for each coordinate
    mae_x = mean_absolute_error(X_true, X_pred)
    mae_y = mean_absolute_error(Y_true, Y_pred)
    mae_z = mean_absolute_error(Z_true, Z_pred)

    # R^2 for each coordinate
    r2_x = r2_score(X_true, X_pred)
    r2_y = r2_score(Y_true, Y_pred)
    r2_z = r2_score(Z_true, Z_pred)

    metrics_dict.update({
        "MSE_X": mse_x,
        "MSE_Y": mse_y,
        "MSE_Z": mse_z,
        "RMSE_X": rmse_x,
        "RMSE_Y": rmse_y,
        "RMSE_Z": rmse_z,
        "MAE_X": mae_x,
        "MAE_Y": mae_y,
        "MAE_Z": mae_z,
        "R2_X": r2_x,
        "R2_Y": r2_y,
        "R2_Z": r2_z
    })

    # -----------------------------
    # 2. 3D displacement metrics
    # -----------------------------
    # Euclidean distance per sample
    distances = np.sqrt((X_pred - X_true)**2 + (Y_pred - Y_true)**2 + (Z_pred - Z_true)**2)

    # Mean distance
    mean_distance = np.mean(distances)
    # Median distance
    median_distance = np.median(distances)
    # Max distance
    max_distance = np.max(distances)

    metrics_dict.update({
        "Mean_3D_Displacement": mean_distance,
        "Median_3D_Displacement": median_distance,
        "Max_3D_Displacement": max_distance
    })

    return metrics_dict


def _export_metrics(metrics_dict, **params):
    """
    - export_path: path to output file. If None, we simply skip saving to CSV/JSON
                  but ALWAYS log them via logger.
                  If it ends with '.csv', we save as CSV.
                  If it ends with '.json', we save as JSON.
                  Otherwise, we skip saving and only log.
    """
    export_path = params.get("export_path")
    logger = params.get("log")

    # Build a string to log (human-readable)
    output = "===== Metrics =====\n"
    for k, v in metrics_dict.items():
        output += f"{k}: {v:.4f}\n"

    # Log metrics to the text log
    logger.info(output)

    # Optionally saving to a file (CSV/JSON) if export_path is provided
    if export_path is not None and len(export_path.strip()) > 0:
        _, file_ext = os.path.splitext(export_path)
        file_ext = file_ext.lower()

        # Save JSON
        if file_ext == ".json":
            with open(export_path, "w") as f:
                json.dump(metrics_dict, f, indent=4)
            logger.info(f"Metrics saved to JSON => {export_path}")

        # Save CSV
        elif file_ext == ".csv":
            with open(export_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for k, v in metrics_dict.items():
                    writer.writerow([k, v])
            logger.info(f"Metrics saved to CSV => {export_path}")

        else:
            # If we don't recognize the extension, we just log a warning
            logger.warning(f"Unrecognized file extension: '{file_ext}'. No file saved.")
    else:
        # If export_path not given, we skip saving but we already logged the metrics above
        logger.info("No export path provided. Metrics not saved to file, only logged.")