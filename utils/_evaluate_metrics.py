import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from fastdtw import fastdtw
import json
import csv
import os

def _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params):
    verbose = params.get("verbose", True)

    if verbose:
        print("Computing enhanced trajectory metrics...")

    X_true = np.array(X_true) if X_true is not None else None
    Y_true = np.array(Y_true) if Y_true is not None else None
    Z_true = np.array(Z_true) if Z_true is not None else None
    X_pred = np.array(X_pred) if X_pred is not None else None
    Y_pred = np.array(Y_pred) if Y_pred is not None else None
    Z_pred = np.array(Z_pred) if Z_pred is not None else None

    metrics_dict = {}

    for coord, true, pred in [('X', X_true, X_pred), 
                            ('Y', Y_true, Y_pred), 
                            ('Z', Z_true, Z_pred)]:
        if true is not None and pred is not None:
            mse = mean_squared_error(true, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true, pred)
            r2 = r2_score(true, pred)
            
            metrics_dict.update({
                f'MSE_{coord}': mse,
                f'RMSE_{coord}': rmse,
                f'MAE_{coord}': mae,
                f'R2_{coord}': r2
            })

    if all(x is not None for x in [X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred]):
        traj_true = np.stack([X_true, Y_true, Z_true], axis=-1)
        traj_pred = np.stack([X_pred, Y_pred, Z_pred], axis=-1)
        
        ate = np.mean(np.sqrt(np.sum((traj_true - traj_pred)**2, axis=-1)))
        
        fte = np.mean(np.sqrt(np.sum((traj_true[:, -1] - traj_pred[:, -1])**2, axis=-1)))
        
        diffs = np.diff(traj_true, axis=0)
        point_distances = np.sqrt(np.sum(diffs**2, axis=1))
        trajectory_length = np.sum(point_distances)
        rte = ate / (trajectory_length + 1e-10)  
              
        corr_x, _ = pearsonr(traj_true[:, 0], traj_pred[:, 0])
        corr_y, _ = pearsonr(traj_true[:, 1], traj_pred[:, 1])
        corr_z, _ = pearsonr(traj_true[:, 2], traj_pred[:, 2])
        temp_corr = np.mean([corr_x, corr_y, corr_z])

        diffs_true = np.diff(traj_true, axis=0)
        diffs_pred = np.diff(traj_pred, axis=0)
        
        true_segments = np.sqrt(np.sum(diffs_true**2, axis=1))
        pred_segments = np.sqrt(np.sum(diffs_pred**2, axis=1))
        
        path_differences = np.abs(true_segments - pred_segments)
        
        mean_path_diff = np.mean(path_differences)
        true_path_length = np.sum(true_segments)
        normalized_path_diff = mean_path_diff / (true_path_length + 1e-10)

        dtw_x, _ = fastdtw(traj_true[:, 0], traj_pred[:, 0])
        dtw_y, _ = fastdtw(traj_true[:, 1], traj_pred[:, 1])
        dtw_z, _ = fastdtw(traj_true[:, 2], traj_pred[:, 2])

        dtw_total = np.sqrt(dtw_x**2 + dtw_y**2 + dtw_z**2)
        dtw_mean = np.mean([dtw_x, dtw_y, dtw_z])

        endpoint_errors = np.sqrt(
            (X_true - X_pred)**2 +
            (Y_true - Y_pred)**2 +
            (Z_true - Z_pred)**2
        )

        distances = np.sqrt((X_pred - X_true)**2 + (Y_pred - Y_true)**2 + (Z_pred - Z_true)**2)

        mean_distance = np.mean(distances)
        median_distance = np.median(distances)
        max_distance = np.max(distances)
        
        metrics_dict.update({
            'Average_Trajectory_Error': ate,
            'Final_Trajectory_Error': fte,
            'Relative_Trajectory_Error': np.mean(rte),
            'Temporal_Correlation': np.mean(temp_corr),
            'DTW_Total': dtw_total,
            'DTW_Mean': dtw_mean,
            'DTW_X': dtw_x,
            'DTW_Y': dtw_y,
            'DTW_Z': dtw_z,
            'Endpoint_Error_Mean': np.mean(endpoint_errors),
            'Endpoint_Error_Std': np.std(endpoint_errors),
            'Path_Difference_Mean': mean_path_diff,
            'Path_Difference_Normalized': normalized_path_diff,
            "Mean_3D_Displacement": mean_distance,
            "Median_3D_Displacement": median_distance,
            "Max_3D_Displacement": max_distance
            })

    return metrics_dict

def _export_metrics(metrics_dict, **params):
    export_path = params.get("export_path")
    logger = params.get("log")

    output = "===== Metrics =====\n"
    output += "\n=== Basic Metrics ===\n"
    basic_metrics = {k: v for k, v in metrics_dict.items() 
                    if any(k.startswith(p) for p in ['MSE_', 'RMSE_', 'MAE_', 'R2_'])}
    for k, v in basic_metrics.items():
        output += f"{k}: {v:.4f}\n"
    
    output += "\n=== Enhanced Trajectory Metrics ===\n"
    enhanced_metrics = {k: v for k, v in metrics_dict.items() 
                       if not any(k.startswith(p) for p in ['MSE_', 'RMSE_', 'MAE_', 'R2_'])}
    for k, v in enhanced_metrics.items():
        output += f"{k}: {v:.4f}\n"

    if logger:
        logger.info(output)

    if export_path is not None and len(export_path.strip()) > 0:
        _, file_ext = os.path.splitext(export_path)
        file_ext = file_ext.lower()

        if file_ext == ".json":
            with open(export_path, "w") as f:
                json.dump(metrics_dict, f, indent=4)
            if logger:
                logger.info(f"Metrics saved to JSON => {export_path}")

        elif file_ext == ".csv":
            with open(export_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for k, v in metrics_dict.items():
                    writer.writerow([k, v])
            if logger:
                logger.info(f"Metrics saved to CSV => {export_path}")

        else:
            if logger:
                logger.warning(f"Unrecognized file extension: '{file_ext}'. No file saved.")
    else:
        if logger:
            logger.info("No export path provided. Metrics not saved to file, only logged.")
