from Preprocessing import _inverse_transform
from utils import _aggregate_sequence_predictions, _save_prediction_results, _plot_loss
from utils._evaluate_metrics import _compute_metrics, _export_metrics

def _train_and_evaluate_model(X_train_Y_coordinate, X_val_Y_coordinate, X_test_Y_coordinate, 
                              y_train_Y_coordinate, y_val_Y_coordinate, y_test_Y_coordinate,
                              X_train_XZ_coordinate, X_val_XZ_coordinate, X_test_XZ_coordinate, 
                              y_train_XZ_coordinate, y_val_XZ_coordinate, y_test_XZ_coordinate, 
                              Y_scaler, XZ_scaler, **params):

    model_Y = params.get('model_Y')
    model_XZ = params.get('model_XZ')
    num_epochs = params.get('num_epochs')
    batch_size = params.get('batch_size')
    coordinates = params.get("coordinates")
    verbose = params.get('verbose')
    # ^ Optionally specify an output path, e.g. "results/metrics.csv" or "results/metrics.json"

    if verbose:
        print("Training the models...")

    # Initialize placeholders for the final ground-truth/pred arrays
    # so we can compute metrics after we do the predictions.
    X_true, Y_true, Z_true = None, None, None
    X_pred, Y_pred, Z_pred = None, None, None

    for coordinate in coordinates:
        if coordinate == "Y":
            # ------------------
            # Train & Predict Y
            # ------------------
            history_Y = model_Y.fit(
                X_train_Y_coordinate,
                y_train_Y_coordinate,
                epochs=num_epochs,
                batch_size=batch_size,
                validation_data=(X_val_Y_coordinate, y_val_Y_coordinate)
            )

            y_pred_Y_coordinate = model_Y.predict(X_test_Y_coordinate)

            # Inverse transform
            y_test_Y_coordinate = _inverse_transform(Y_scaler, y_test_Y_coordinate, coordinate, **params)
            y_pred_Y_coordinate = _inverse_transform(Y_scaler, y_pred_Y_coordinate, coordinate, **params)

            # Aggregate sequences
            y_test_Y_coordinate = _aggregate_sequence_predictions(y_test_Y_coordinate, **params)
            y_pred_aggregated = _aggregate_sequence_predictions(y_pred_Y_coordinate, **params)

            Y_true = y_test_Y_coordinate[:, 0]
            Y_pred = y_pred_aggregated[:, 0]

            _plot_loss(history_Y, coordinate, **params)

        if coordinate == "XZ":
            # ------------------
            # Train & Predict XZ
            # ------------------
            history_XZ = model_XZ.fit(
                X_train_XZ_coordinate,
                y_train_XZ_coordinate,
                epochs=num_epochs,
                batch_size=batch_size,
                validation_data=(X_val_XZ_coordinate, y_val_XZ_coordinate)
            )

            y_pred_XZ_coordinate = model_XZ.predict(X_test_XZ_coordinate)

            # Inverse transform
            y_test_XZ_coordinate = _inverse_transform(XZ_scaler, y_test_XZ_coordinate, coordinate, **params)
            y_pred_XZ_coordinate = _inverse_transform(XZ_scaler, y_pred_XZ_coordinate, coordinate, **params)

            # Aggregate
            y_test_XZ_coordinate = _aggregate_sequence_predictions(y_test_XZ_coordinate, **params)
            y_pred_aggregated = _aggregate_sequence_predictions(y_pred_XZ_coordinate, **params)

            # Split aggregated XZ into X and Z
            X_true = y_test_XZ_coordinate[:, 0]
            Z_true = y_test_XZ_coordinate[:, 1]
            X_pred = y_pred_aggregated[:, 0]
            Z_pred = y_pred_aggregated[:, 1]

            _plot_loss(history_XZ, coordinate, **params)

    # ----------------
    # Save predictions
    # ----------------
    # Here, X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred are from the final test sets.
    _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params)

    # ----------------
    # Compute metrics
    # ----------------
    if verbose:
        print("Computing metrics...")

    metrics_dict = _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params)

    # --------------------------------
    # Export or log the metrics
    # --------------------------------
    _export_metrics(metrics_dict, **params)

    # You may also want to return metrics_dict for further usage:
    return metrics_dict