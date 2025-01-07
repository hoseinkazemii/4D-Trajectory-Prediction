from Preprocessing import _inverse_transform
from utils import _aggregate_sequence_predictions, _save_prediction_results, _plot_loss
from utils._evaluate_metrics import _compute_metrics, _export_metrics

def _train_and_evaluate_model(split_data_dict, scalers_dict, **params):
    """
    Dynamically trains each model in 'models_dict' using data from 'split_data_dict',
    then inversely transforms and aggregates predictions for each coordinate group.

    models_dict: dict of { coordinate_str: Keras Model }
      e.g. { "Y": model_Y, "XZ": model_XZ, ... }

    split_data_dict: dict of { coordinate_str: dict(...) }
      Each coordinate_str maps to:
        {
          "X_train": np.array,
          "y_train": np.array,
          "X_val":   np.array,
          "y_val":   np.array,
          "X_test":  np.array,
          "y_test":  np.array
        }

    scalers_dict: dict of { coordinate_str: scaler_object }
      e.g. { "Y": Y_scaler, "XZ": XZ_scaler, ... }
    """

    coordinates = params.get("coordinates")
    verbose = params.get('verbose', True)
    num_epochs = params.get('num_epochs')
    batch_size = params.get('batch_size')
    models_dict = params.get("models_dict")

    if verbose:
        print("Training the models for coordinates:", coordinates)
        print("Transforming predictions back to coordinates...")
        print("Aggregating sequence predictions...")

    # Initialize placeholders for final combined results
    X_true, Y_true, Z_true = None, None, None
    X_pred, Y_pred, Z_pred = None, None, None

    # A helper function to fill X_true, Y_true, Z_true, etc. from a multi-dimensional array
    def _assign_results(y_true_array, y_pred_array, coord_str):
        """
        y_true_array, y_pred_array have shape (num_samples, prediction_horizon, dimension).
        For example, if coord_str == "XZ", dimension=2 => columns [0=>X, 1=>Z].
        If coord_str == "XYZ", dimension=3 => columns [0=>X, 1=>Y, 2=>Z].
        We'll assign global X_true, Y_true, Z_true, etc. accordingly.
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
            # If we have some other combination, raise an error or handle accordingly
            raise ValueError(f"Unexpected coordinate string '{coord_str}'")

    # Train & Predict each coordinate group
    for coord_str in coordinates:
        model = models_dict[coord_str]
        scaler = scalers_dict[coord_str]

        # Retrieve train/val/test arrays
        train_X = split_data_dict[coord_str]["X_train"]
        train_y = split_data_dict[coord_str]["y_train"]
        val_X   = split_data_dict[coord_str]["X_val"]
        val_y   = split_data_dict[coord_str]["y_val"]
        test_X  = split_data_dict[coord_str]["X_test"]
        test_y  = split_data_dict[coord_str]["y_test"]

        # Fit
        history = model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=1 if verbose else 0
        )

        # Predict on test
        y_pred_test = model.predict(test_X)

        # Inverse transform (test_y and y_pred_test)
        # shape => (num_samples, prediction_horizon, dimension_of(coord_str))
        y_true_inv = _inverse_transform(scaler, test_y, coord_str, **params)
        y_pred_inv = _inverse_transform(scaler, y_pred_test, coord_str, **params)

        # Aggregate sequences
        y_true_agg = _aggregate_sequence_predictions(y_true_inv, **params)
        y_pred_agg = _aggregate_sequence_predictions(y_pred_inv, **params)

        # Assign results to X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred
        _assign_results(y_true_agg, y_pred_agg, coord_str)

        # Plot loss for each coordinate group
        _plot_loss(history, coord_str, **params)

    # Now, we have final X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred
    # depending on which coordinates exist. Let's save them:
    _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params)

    if verbose:
        print("Computing metrics...")

    # Compute metrics (handles partial or full if some are None)
    metrics_dict = _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params)

    # Export or log metrics
    _export_metrics(metrics_dict, **params)

    return metrics_dict
