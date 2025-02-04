from Preprocessing import _inverse_transform
from utils import _aggregate_sequence_predictions, _save_prediction_results, _plot_loss
from utils._evaluate_metrics import _compute_metrics, _export_metrics
import pandas as pd

def _train_and_evaluate_model(split_data_dict, scalers_dict, row_counts, **params):
    """
    Trains each GRU model in 'models_dict' using data from 'split_data_dict' then performs
    inverse transformation and aggregation of the predictions.
    
    split_data_dict: dict of { coordinate_str: dict with keys X_train, y_train, X_val, y_val, X_test, y_test }
    scalers_dict: dict of { coordinate_str: scaler_object }
    """
    coordinates = params.get("coordinates")
    verbose = params.get('verbose', True)
    num_epochs = params.get('num_epochs')
    batch_size = params.get('batch_size')
    models_dict = params.get("models_dict")

    if verbose:
        print("Training the GRU models for coordinates:", coordinates)
        print("Transforming predictions back to coordinates...")
        print("Aggregating sequence predictions...")

    # Initialize placeholders for final combined results
    X_true, Y_true, Z_true = None, None, None
    X_pred, Y_pred, Z_pred = None, None, None

    def _assign_results(y_true_array, y_pred_array, coord_str):
        """
        Given y_true_array and y_pred_array (shape: [num_samples, prediction_horizon, dim]),
        assign the appropriate columns to global variables for X, Y, and Z.
        """
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

    # For each coordinate group, train and then predict
    for coord_str in coordinates:
        model = models_dict[coord_str]
        scaler = scalers_dict[coord_str]
        # Retrieve the training/validation/test data arrays
        train_X = split_data_dict[coord_str]["X_train"]
        train_y = split_data_dict[coord_str]["y_train"]
        val_X = split_data_dict[coord_str]["X_val"]
        val_y = split_data_dict[coord_str]["y_val"]
        test_X = split_data_dict[coord_str]["X_test"]
        test_y = split_data_dict[coord_str]["y_test"]

        # Fit the model
        history = model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=1 if verbose else 0
        )

        # Predict on test data
        y_pred_test = model.predict(test_X)

        # Inverse transform the scaled predictions and ground truth values
        y_true_inv = _inverse_transform(scaler, test_y, coord_str, **params)
        y_pred_inv = _inverse_transform(scaler, y_pred_test, coord_str, **params)

        # Aggregate the sequence predictions if needed
        y_true_agg = _aggregate_sequence_predictions(y_true_inv, row_counts, **params)
        y_pred_agg = _aggregate_sequence_predictions(y_pred_inv, row_counts, **params)

        # Assign the results to the appropriate coordinate arrays
        _assign_results(y_true_agg, y_pred_agg, coord_str)

        # Optionally plot the training loss
        _plot_loss(history, coord_str, **params)

    # Save the prediction results and compute metrics
    _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, row_counts, **params)
    metrics_dict = _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params)
    _export_metrics(metrics_dict, **params)
    return metrics_dict
