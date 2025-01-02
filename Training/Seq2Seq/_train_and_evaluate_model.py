from Preprocessing import _inverse_transform
from utils import _aggregate_sequence_predictions, _save_prediction_results, _plot_loss

def _train_and_evaluate_model(X_train_Y_coordinate, X_val_Y_coordinate, X_test_Y_coordinate, y_train_Y_coordinate, y_val_Y_coordinate, y_test_Y_coordinate, \
                              X_train_XZ_coordinate, X_val_XZ_coordinate, X_test_XZ_coordinate, y_train_XZ_coordinate, y_val_XZ_coordinate, y_test_XZ_coordinate, \
                              Y_scaler, XZ_scaler, **params):
    model_Y = params.get('model_Y')
    model_XZ = params.get('model_XZ')
    num_epochs = params.get('num_epochs')
    batch_size = params.get('batch_size')
    coordinates = params.get("coordinates")
    verbose = params.get('verbose')
    if verbose:
        print("Training the models...")
        print("Transforming predictions back to coordinates...")
        print("Aggregating sequence predictions...")

    for coordinate in coordinates:
        if coordinate == "Y":
            history_Y = model_Y.fit(X_train_Y_coordinate, y_train_Y_coordinate, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val_Y_coordinate, y_val_Y_coordinate))
            y_pred_Y_coordinate = model_Y.predict(X_test_Y_coordinate)
            y_test_Y_coordinate = _inverse_transform(Y_scaler, y_test_Y_coordinate, coordinate, **params)
            y_pred_Y_coordinate = _inverse_transform(Y_scaler, y_pred_Y_coordinate, coordinate, **params)
            y_test_Y_coordinate = _aggregate_sequence_predictions(y_test_Y_coordinate, **params)
            y_pred_aggregated = _aggregate_sequence_predictions(y_pred_Y_coordinate, **params)
            Y_true = y_test_Y_coordinate[:, 0]
            Y_pred = y_pred_aggregated[:, 0]
            _plot_loss(history_Y, coordinate, **params)
        if coordinate == "XZ":
            history_XZ = model_XZ.fit(X_train_XZ_coordinate, y_train_XZ_coordinate, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val_XZ_coordinate, y_val_XZ_coordinate))
            y_pred_XZ_coordinate = model_XZ.predict(X_test_XZ_coordinate)
            y_test_XZ_coordinate = _inverse_transform(XZ_scaler, y_test_XZ_coordinate, coordinate, **params)
            y_pred_XZ_coordinate = _inverse_transform(XZ_scaler, y_pred_XZ_coordinate, coordinate, **params)
            y_test_XZ_coordinate = _aggregate_sequence_predictions(y_test_XZ_coordinate, **params)
            y_pred_aggregated = _aggregate_sequence_predictions(y_pred_XZ_coordinate, **params)
            # Split aggregated XZ into X and Z
            X_true = y_test_XZ_coordinate[:, 0]
            Z_true = y_test_XZ_coordinate[:, 1]
            X_pred = y_pred_aggregated[:, 0]
            Z_pred = y_pred_aggregated[:, 1]
            _plot_loss(history_XZ, coordinate, **params)

    _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params)
