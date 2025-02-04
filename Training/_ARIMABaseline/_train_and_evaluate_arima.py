# _train_and_evaluate_arima.py
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA

# Assume that these helper functions are available in your codebase.
from Preprocessing import _inverse_transform
from utils import _aggregate_sequence_predictions, _save_prediction_results
from utils._evaluate_metrics import _compute_metrics, _export_metrics

def _train_and_evaluate_arima(split_data_dict, scalers_dict, row_counts, **params):
    """
    For each coordinate group (e.g. "X", "YZ", "XYZ", etc.) this function:
      1. Reconstructs a full training series (per univariate dimension) from the training set.
      2. Trains ARIMA models (one per coordinate dimension) using only the training data.
      3. For each sample in the validation and test sets, uses the input window to update
         the pre-trained ARIMA model (via .apply) and then forecasts the next prediction_horizon points.
      4. Inversely transforms and aggregates the predictions for fair comparison.
    
    It is assumed that the training (and validation/test) arrays were generated as non-overlapping sliding windows
    from a single continuous time series.
    
    Parameters:
      - split_data_dict: dictionary with keys: "X_train", "y_train", "X_val", "y_val", "X_test", "y_test"
      - scalers_dict: dictionary mapping coordinate group to its scaler object
      - row_counts: (for aggregation, as in your other modules)
      - params: additional parameters (must include "coordinates", "sequence_length", "prediction_horizon", "arima_order", etc.)
    """
    coordinates = params.get("coordinates")
    verbose = params.get("verbose", True)
    arima_order = params.get("arima_order")
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")

    # Final aggregated results (to be used by your utilities)
    X_true, Y_true, Z_true = None, None, None
    X_pred, Y_pred, Z_pred = None, None, None

    # Loop over each coordinate group (e.g., "X", "YZ", "XYZ", "XZ", etc.)
    for coord_str in coordinates:
        scaler = scalers_dict[coord_str]
        # Retrieve the sliding-window data for this coordinate group.
        X_train = split_data_dict[coord_str]["X_train"]  # shape: (n_train, sequence_length, d)
        y_train = split_data_dict[coord_str]["y_train"]  # shape: (n_train, prediction_horizon, d)
        X_val   = split_data_dict[coord_str]["X_val"]
        y_val   = split_data_dict[coord_str]["y_val"]
        X_test  = split_data_dict[coord_str]["X_test"]
        y_test  = split_data_dict[coord_str]["y_test"]

        n_train, seq_len, d = X_train.shape

        if verbose:
            print(f"Building ARIMA baseline for coordinate group '{coord_str}' with {d} dimension(s).")

        # === Step 1. Reassemble the full training series for each dimension ===
        # Here we assume that the training samples are non-overlapping.
        # For each dimension, the full training series is:
        #   initial_window (from the first sample) concatenated with each sample's y_train.
        full_train_series = {}
        for dim in range(d):
            # Start with the entire input window of the first sample.
            train_series = list(X_train[0, :, dim])
            # Append each training sample’s forecast (assumed non-overlapping).
            for i in range(n_train):
                train_series.extend(y_train[i, :, dim])
            full_train_series[dim] = np.array(train_series)

        # === Step 2. Train ARIMA on the training series (per dimension) ===
        fitted_models = {}
        for dim in range(d):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model = ARIMA(full_train_series[dim], order=arima_order)
                    fitted_model = model.fit()
                    if verbose:
                        print(f"Trained ARIMA for coord '{coord_str}' dimension {dim} (order={arima_order}).")
                except Exception as e:
                    if verbose:
                        print(f"ARIMA training failed for coord '{coord_str}' dimension {dim}: {e}")
                    fitted_model = None
            fitted_models[dim] = fitted_model

        # === Step 3. Define a helper to forecast for a given set (validation or test) ===
        def forecast_for_set(X_set, set_name=""):
            """
            For each sample in X_set (shape: [n_samples, sequence_length, d]),
            update the pre-trained ARIMA (for each dimension) with the sample’s input window and forecast.
            Returns an array of forecasts with shape (n_samples, prediction_horizon, d).
            """
            n_samples = X_set.shape[0]
            forecasts = np.zeros((n_samples, prediction_horizon, d))
            for i in range(n_samples):
                for dim in range(d):
                    if fitted_models[dim] is None:
                        # Fallback: use the last value of the input window.
                        window = X_set[i, :, dim]
                        forecast = np.repeat(window[-1], prediction_horizon)
                    else:
                        window = X_set[i, :, dim]
                        try:
                            # Update the trained model’s state with the new input window.
                            updated_model = fitted_models[dim].apply(window)
                            forecast = updated_model.forecast(steps=prediction_horizon)
                        except Exception as e:
                            if verbose:
                                print(f"ARIMA forecasting failed for coord '{coord_str}', sample {i}, dimension {dim} on {set_name}: {e}")
                            forecast = np.repeat(window[-1], prediction_horizon)
                    forecasts[i, :, dim] = forecast
            return forecasts

        # Forecast for the validation and test sets.
        # (If you wish to report validation performance as well, you can store those forecasts.)
        y_val_pred = forecast_for_set(X_val, set_name="validation")
        y_test_pred = forecast_for_set(X_test, set_name="test")

        # === Step 4. Inverse transform and aggregate the predictions ===
        # (Your utility functions expect arrays shaped like (num_samples, prediction_horizon, dimension).)
        y_val_true_inv  = _inverse_transform(scaler, y_val, coord_str, **params)
        y_val_pred_inv  = _inverse_transform(scaler, y_val_pred, coord_str, **params)
        y_test_true_inv = _inverse_transform(scaler, y_test, coord_str, **params)
        y_test_pred_inv = _inverse_transform(scaler, y_test_pred, coord_str, **params)

        y_val_true_agg  = _aggregate_sequence_predictions(y_val_true_inv, row_counts, **params)
        y_val_pred_agg  = _aggregate_sequence_predictions(y_val_pred_inv, row_counts, **params)
        y_test_true_agg = _aggregate_sequence_predictions(y_test_true_inv, row_counts, **params)
        y_test_pred_agg = _aggregate_sequence_predictions(y_test_pred_inv, row_counts, **params)

        # For fair comparison we will use the test-set results as the final evaluation.
        def _assign_results(y_true_array, y_pred_array, coord_str):
            nonlocal X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred
            if coord_str == "X":
                X_true = y_true_array[:, 0]
                X_pred = y_pred_array[:, 0]
            elif coord_str == "Y":
                Y_true = y_true_array[:, 0]
                Y_pred = y_pred_array[:, 0]
            elif coord_str == "Z":
                Z_true = y_true_array[:, 0]
                Z_pred = y_pred_array[:, 0]
            elif coord_str == "XY":
                X_true = y_true_array[:, 0]
                Y_true = y_true_array[:, 1]
                X_pred = y_pred_array[:, 0]
                Y_pred = y_pred_array[:, 1]
            elif coord_str == "XZ":
                X_true = y_true_array[:, 0]
                Z_true = y_true_array[:, 1]
                X_pred = y_pred_array[:, 0]
                Z_pred = y_pred_array[:, 1]
            elif coord_str == "YZ":
                Y_true = y_true_array[:, 0]
                Z_true = y_true_array[:, 1]
                Y_pred = y_pred_array[:, 0]
                Z_pred = y_pred_array[:, 1]
            elif coord_str == "XYZ":
                X_true = y_true_array[:, 0]
                Y_true = y_true_array[:, 1]
                Z_true = y_true_array[:, 2]
                X_pred = y_pred_array[:, 0]
                Y_pred = y_pred_array[:, 1]
                Z_pred = y_pred_array[:, 2]
            else:
                raise ValueError(f"Unexpected coordinate string '{coord_str}'")

        _assign_results(y_test_true_agg, y_test_pred_agg, coord_str)

        if verbose:
            print(f"ARIMA baseline for coordinate group '{coord_str}' completed (validation and test forecasting).")

    # === Final: Save results and compute metrics ===
    _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, row_counts, **params)
    metrics_dict = _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params)
    _export_metrics(metrics_dict, **params)
    return metrics_dict
