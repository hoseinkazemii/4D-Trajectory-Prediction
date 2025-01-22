import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from dtaidistance import dtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import json
import csv
import os

def _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params):
    """
    Computes comprehensive metrics for 3D trajectory prediction tasks.
    Maintains compatibility with existing logger structure.
    
    Parameters:
    -----------
    X_true, Y_true, Z_true : array-like
        Ground truth coordinates
    X_pred, Y_pred, Z_pred : array-like
        Predicted coordinates
    params : dict
        Additional parameters including:
        - verbose: bool, whether to print progress
        - batch_first: bool, whether the first dimension is batch
        
    Returns:
    --------
    dict
        Dictionary containing all computed metrics
    """
    verbose = params.get("verbose", True)

    if verbose:
        print("Computing enhanced trajectory metrics...")

    # Convert to numpy arrays and ensure correct shape
    X_true = np.array(X_true) if X_true is not None else None
    Y_true = np.array(Y_true) if Y_true is not None else None
    Z_true = np.array(Z_true) if Z_true is not None else None
    X_pred = np.array(X_pred) if X_pred is not None else None
    Y_pred = np.array(Y_pred) if Y_pred is not None else None
    Z_pred = np.array(Z_pred) if Z_pred is not None else None

    metrics_dict = {}

    # 1. Basic Coordinate-wise Metrics (Original metrics)
    # ------------------------------------------------
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

    # 2. Enhanced Trajectory Metrics
    # ----------------------------
    # Only compute if we have all coordinates
    if all(x is not None for x in [X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred]):
        # Combine coordinates into trajectories
        traj_true = np.stack([X_true, Y_true, Z_true], axis=-1)
        traj_pred = np.stack([X_pred, Y_pred, Z_pred], axis=-1)
        
        # Average trajectory error (ATE)
        ate = np.mean(np.sqrt(np.sum((traj_true - traj_pred)**2, axis=-1)))
        
        # Final trajectory error (FTE)
        fte = np.mean(np.sqrt(np.sum((traj_true[:, -1] - traj_pred[:, -1])**2, axis=-1)))
        
        # Relative trajectory error (RTE)
        diffs = np.diff(traj_true, axis=0)  # Shape: (97, 3)
        point_distances = np.sqrt(np.sum(diffs**2, axis=1))  # Shape: (97,)
        trajectory_length = np.sum(point_distances)
        rte = ate / (trajectory_length + 1e-10)  
              
        # Temporal correlation
        # Calculate correlation for each coordinate independently
        # Extract X, Y, Z columns (all samples)
        corr_x, _ = pearsonr(traj_true[:, 0], traj_pred[:, 0])  # X coordinates
        corr_y, _ = pearsonr(traj_true[:, 1], traj_pred[:, 1])  # Y coordinates
        corr_z, _ = pearsonr(traj_true[:, 2], traj_pred[:, 2])  # Z coordinates
        temp_corr = np.mean([corr_x, corr_y, corr_z])

        # Calculate path length difference between true and predicted trajectories
        diffs_true = np.diff(traj_true, axis=0)
        diffs_pred = np.diff(traj_pred, axis=0)
        
        # Calculate segment lengths for both trajectories
        true_segments = np.sqrt(np.sum(diffs_true**2, axis=1))
        pred_segments = np.sqrt(np.sum(diffs_pred**2, axis=1))
        
        # Calculate absolute difference between corresponding segments
        path_differences = np.abs(true_segments - pred_segments)
        
        # Calculate mean path difference and normalize by true path length
        mean_path_diff = np.mean(path_differences)
        true_path_length = np.sum(true_segments)
        normalized_path_diff = mean_path_diff / (true_path_length + 1e-10)


        # DTW metrics - calculate separately for each dimension
        dtw_x, _ = fastdtw(traj_true[:, 0], traj_pred[:, 0])  # X coordinates
        dtw_y, _ = fastdtw(traj_true[:, 1], traj_pred[:, 1])  # Y coordinates
        dtw_z, _ = fastdtw(traj_true[:, 2], traj_pred[:, 2])  # Z coordinates
        
        # Combine DTW distances
        dtw_total = np.sqrt(dtw_x**2 + dtw_y**2 + dtw_z**2)  # Euclidean combination
        dtw_mean = np.mean([dtw_x, dtw_y, dtw_z])  # Average of individual dimensions
        
        # Endpoint errors - using final values
        endpoint_errors = np.sqrt(
            (X_true - X_pred)**2 +
            (Y_true - Y_pred)**2 +
            (Z_true - Z_pred)**2
        )

        # -----------------------------
        # 3D displacement metrics
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
            'Average_Trajectory_Error': ate,
            'Final_Trajectory_Error': fte,
            'Relative_Trajectory_Error': np.mean(rte),
            'Temporal_Correlation': np.mean(temp_corr),
            'DTW_Total': dtw_total,  # Combined DTW distance
            'DTW_Mean': dtw_mean,  # Average of individual dimensions
            'DTW_X': dtw_x,  # Individual dimension DTW
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
    """
    Exports metrics to file and logs them via logger.
    Maintains compatibility with existing logger structure.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of computed metrics
    params : dict
        Additional parameters including:
        - export_path: path to output file (optional)
        - log: logger object
        
    If export_path ends with:
        - '.csv': save as CSV
        - '.json': save as JSON
        Otherwise, only log metrics
    """
    export_path = params.get("export_path")
    logger = params.get("log")

    # Build a string to log (human-readable)
    output = "===== Metrics =====\n"
    output += "\n=== Basic Metrics ===\n"
    # First log basic metrics
    basic_metrics = {k: v for k, v in metrics_dict.items() 
                    if any(k.startswith(p) for p in ['MSE_', 'RMSE_', 'MAE_', 'R2_'])}
    for k, v in basic_metrics.items():
        output += f"{k}: {v:.4f}\n"
    
    # Then log enhanced metrics
    output += "\n=== Enhanced Trajectory Metrics ===\n"
    enhanced_metrics = {k: v for k, v in metrics_dict.items() 
                       if not any(k.startswith(p) for p in ['MSE_', 'RMSE_', 'MAE_', 'R2_'])}
    for k, v in enhanced_metrics.items():
        output += f"{k}: {v:.4f}\n"

    # Log metrics to the text log
    if logger:
        logger.info(output)

    # Optionally saving to a file (CSV/JSON) if export_path is provided
    if export_path is not None and len(export_path.strip()) > 0:
        _, file_ext = os.path.splitext(export_path)
        file_ext = file_ext.lower()

        # Save JSON
        if file_ext == ".json":
            with open(export_path, "w") as f:
                json.dump(metrics_dict, f, indent=4)
            if logger:
                logger.info(f"Metrics saved to JSON => {export_path}")

        # Save CSV
        elif file_ext == ".csv":
            with open(export_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for k, v in metrics_dict.items():
                    writer.writerow([k, v])
            if logger:
                logger.info(f"Metrics saved to CSV => {export_path}")

        else:
            # If we don't recognize the extension, we just log a warning
            if logger:
                logger.warning(f"Unrecognized file extension: '{file_ext}'. No file saved.")
    else:
        # If export_path not provided, we skip saving but we already logged the metrics above
        if logger:
            logger.info("No export path provided. Metrics not saved to file, only logged.")




#########################################
# import json
# import csv
# import os
# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# def _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params):
#     """
#     Computes a variety of metrics for 3D regression tasks.
#     Returns a dictionary of metric_name -> value.
#     """
#     verbose = params.get("verbose")

#     if verbose:
#         print("Computing metrics...")
#     metrics_dict = {}

#     # Convert to numpy arrays (if they aren't already)
#     X_true = np.array(X_true)
#     Y_true = np.array(Y_true)
#     Z_true = np.array(Z_true)
#     X_pred = np.array(X_pred)
#     Y_pred = np.array(Y_pred)
#     Z_pred = np.array(Z_pred)

#     # -----------------------------
#     # 1. Coordinate-wise metrics
#     # -----------------------------
#     # MSE for each coordinate
#     mse_x = mean_squared_error(X_true, X_pred)
#     mse_y = mean_squared_error(Y_true, Y_pred)
#     mse_z = mean_squared_error(Z_true, Z_pred)

#     # RMSE for each coordinate
#     rmse_x = np.sqrt(mse_x)
#     rmse_y = np.sqrt(mse_y)
#     rmse_z = np.sqrt(mse_z)

#     # MAE for each coordinate
#     mae_x = mean_absolute_error(X_true, X_pred)
#     mae_y = mean_absolute_error(Y_true, Y_pred)
#     mae_z = mean_absolute_error(Z_true, Z_pred)

#     # R^2 for each coordinate
#     r2_x = r2_score(X_true, X_pred)
#     r2_y = r2_score(Y_true, Y_pred)
#     r2_z = r2_score(Z_true, Z_pred)

#     metrics_dict.update({
#         "MSE_X": mse_x,
#         "MSE_Y": mse_y,
#         "MSE_Z": mse_z,
#         "RMSE_X": rmse_x,
#         "RMSE_Y": rmse_y,
#         "RMSE_Z": rmse_z,
#         "MAE_X": mae_x,
#         "MAE_Y": mae_y,
#         "MAE_Z": mae_z,
#         "R2_X": r2_x,
#         "R2_Y": r2_y,
#         "R2_Z": r2_z
#     })

#     # -----------------------------
#     # 2. 3D displacement metrics
#     # -----------------------------
#     # Euclidean distance per sample
#     distances = np.sqrt((X_pred - X_true)**2 + (Y_pred - Y_true)**2 + (Z_pred - Z_true)**2)

#     # Mean distance
#     mean_distance = np.mean(distances)
#     # Median distance
#     median_distance = np.median(distances)
#     # Max distance
#     max_distance = np.max(distances)

#     metrics_dict.update({
#         "Mean_3D_Displacement": mean_distance,
#         "Median_3D_Displacement": median_distance,
#         "Max_3D_Displacement": max_distance
#     })

#     return metrics_dict


# def _export_metrics(metrics_dict, **params):
#     """
#     - export_path: path to output file. If None, we simply skip saving to CSV/JSON
#                   but ALWAYS log them via logger.
#                   If it ends with '.csv', we save as CSV.
#                   If it ends with '.json', we save as JSON.
#                   Otherwise, we skip saving and only log.
#     """
#     export_path = params.get("export_path")
#     logger = params.get("log")

#     # Build a string to log (human-readable)
#     output = "===== Metrics =====\n"
#     for k, v in metrics_dict.items():
#         output += f"{k}: {v:.4f}\n"

#     # Log metrics to the text log
#     logger.info(output)

#     # Optionally saving to a file (CSV/JSON) if export_path is provided
#     if export_path is not None and len(export_path.strip()) > 0:
#         _, file_ext = os.path.splitext(export_path)
#         file_ext = file_ext.lower()

#         # Save JSON
#         if file_ext == ".json":
#             with open(export_path, "w") as f:
#                 json.dump(metrics_dict, f, indent=4)
#             logger.info(f"Metrics saved to JSON => {export_path}")

#         # Save CSV
#         elif file_ext == ".csv":
#             with open(export_path, "w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["Metric", "Value"])
#                 for k, v in metrics_dict.items():
#                     writer.writerow([k, v])
#             logger.info(f"Metrics saved to CSV => {export_path}")

#         else:
#             # If we don't recognize the extension, we just log a warning
#             logger.warning(f"Unrecognized file extension: '{file_ext}'. No file saved.")
#     else:
#         # If export_path not given, we skip saving but we already logged the metrics above
#         logger.info("No export path provided. Metrics not saved to file, only logged.")