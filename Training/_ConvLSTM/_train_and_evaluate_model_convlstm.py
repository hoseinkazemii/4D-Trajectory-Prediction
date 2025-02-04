from Preprocessing import _inverse_transform
from utils import _aggregate_sequence_predictions, _save_prediction_results, _plot_loss
from utils._evaluate_metrics import _compute_metrics, _export_metrics
import numpy as np

def _train_and_evaluate_model_convlstm(split_data_dict, scalers_dict, row_counts, **params):
    """
    For each coordinate group, trains the ConvLSTM model, then predicts, inverse transforms,
    aggregates sequence predictions, and computes/export metrics.

    IMPORTANT: Because our ConvLSTM network expects inputs of shape 
       (batch, sequence_length, 1, in_dim, 1),
    we reshape the training/validation/test inputs accordingly.
    """
    coordinates = params.get("coordinates")
    verbose = params.get('verbose', True)
    num_epochs = params.get('num_epochs')
    batch_size = params.get('batch_size')
    models_dict = params.get("models_dict")

    if verbose:
        print("Training ConvLSTM models for coordinates:", coordinates)
        print("Transforming predictions back to coordinates...")
        print("Aggregating sequence predictions...")

    # Placeholders for final combined results (depending on which coordinate groups you use)
    X_true, Y_true, Z_true = None, None, None
    X_pred, Y_pred, Z_pred = None, None, None

    # Helper to assign results based on coordinate string.
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

    # Helper function to reshape input for ConvLSTM:
    # From (samples, sequence_length, in_dim) to (samples, sequence_length, 1, in_dim, 1)
    def reshape_input(x, in_dim):
        return x.reshape((x.shape[0], x.shape[1], 1, in_dim, 1))

    for coord_str in coordinates:
        model = models_dict[coord_str]
        scaler = scalers_dict[coord_str]
        in_dim = params.get("coord_to_dim")[coord_str]

        # Retrieve train/validation/test arrays.
        train_X = split_data_dict[coord_str]["X_train"]
        train_y = split_data_dict[coord_str]["y_train"]
        val_X   = split_data_dict[coord_str]["X_val"]
        val_y   = split_data_dict[coord_str]["y_val"]
        test_X  = split_data_dict[coord_str]["X_test"]
        test_y  = split_data_dict[coord_str]["y_test"]

        # Reshape inputs for ConvLSTM (the targets remain the same shape).
        train_X = reshape_input(train_X, in_dim)
        val_X = reshape_input(val_X, in_dim)
        test_X = reshape_input(test_X, in_dim)

        # Train the model.
        history = model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=1 if verbose else 0
        )

        # Predict on test data.
        y_pred_test = model.predict(test_X)

        # Inverse transform predictions and targets.
        y_true_inv = _inverse_transform(scaler, test_y, coord_str, **params)
        y_pred_inv = _inverse_transform(scaler, y_pred_test, coord_str, **params)

        # Aggregate sequence predictions.
        y_true_agg = _aggregate_sequence_predictions(y_true_inv, row_counts, **params)
        y_pred_agg = _aggregate_sequence_predictions(y_pred_inv, row_counts, **params)

        # Assign results to global variables (e.g., X_true, Y_true, Z_true, etc.)
        _assign_results(y_true_agg, y_pred_agg, coord_str)

        # Optionally, plot the loss history.
        _plot_loss(history, coord_str, **params)

    # Save final prediction results.
    _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, row_counts, **params)

    # Compute metrics (this function should handle partial or full sets if some are None).
    metrics_dict = _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params)
    _export_metrics(metrics_dict, **params)

    return metrics_dict
