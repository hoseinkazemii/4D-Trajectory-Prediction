import time
import numpy as np
from Preprocessing import _inverse_transform
from utils import _aggregate_sequence_predictions, _save_prediction_results, _plot_loss
from utils._evaluate_metrics import _compute_metrics, _export_metrics
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from ._monte_carlo_dropout_predict import _monte_carlo_dropout_predict

def _train_and_evaluate_model_tcn(split_data_dict, scalers_dict, row_counts, **params):
    coordinates = params.get("coordinates")
    verbose = params.get('verbose', True)
    num_epochs = params.get('num_epochs')
    batch_size = params.get('batch_size')
    models_dict = params.get("models_dict")
    model_name = params.get("model_name")
    use_mc_dropout = params.get("use_mc_dropout")

    if verbose:
        print("Training the models for coordinates:", coordinates)
        print("Transforming predictions back to coordinates...")
        print("Aggregating sequence predictions...")

    X_true, Y_true, Z_true = None, None, None
    X_pred, Y_pred, Z_pred = None, None, None

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

    for coord_str in coordinates:
        model = models_dict[coord_str]
        scaler = scalers_dict[coord_str]

        train_X = split_data_dict[coord_str]["X_train"]
        train_y = split_data_dict[coord_str]["y_train"]
        val_X   = split_data_dict[coord_str]["X_val"]
        val_y   = split_data_dict[coord_str]["y_val"]
        test_X  = split_data_dict[coord_str]["X_test"]
        test_y  = split_data_dict[coord_str]["y_test"]

        model_checkpoint = ModelCheckpoint(
            f'/BestModels/best_{model_name}_model.keras',
            monitor='val_loss',
            save_best_only=True
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=0.0001,
            restore_best_weights=True
        )

        with tf.device('/GPU:0'):
            history = model.fit(
                train_X, train_y,
                validation_data=(val_X, val_y),
                epochs=num_epochs,
                batch_size=batch_size,
                verbose=1 if verbose else 0
            )

        if use_mc_dropout:
            all_pass_preds = _monte_carlo_dropout_predict(model, test_X, **params).numpy()
            pass_mean_displacements = []
            avg_preds = np.mean(all_pass_preds, axis=0)
            y_true_inv = _inverse_transform(scaler, test_y, coord_str, **params)
            y_true_agg = _aggregate_sequence_predictions(y_true_inv, row_counts, test_mode=True, **params)
            mc_dropout_passes = all_pass_preds.shape[0]
            for i in range(mc_dropout_passes):
                pass_pred_i = all_pass_preds[i]
                pass_pred_i_inv = _inverse_transform(scaler, pass_pred_i, coord_str, **params)
                y_pred_agg = _aggregate_sequence_predictions(pass_pred_i_inv, row_counts, test_mode=True, **params)
                if coord_str == "X":
                    dist = np.sqrt((y_pred_agg[:, 0] - y_true_agg[:, 0])**2)
                elif coord_str == "Y":
                    dist = np.sqrt((y_pred_agg[:, 0] - y_true_agg[:, 0])**2)
                elif coord_str == "Z":
                    dist = np.sqrt((y_pred_agg[:, 0] - y_true_agg[:, 0])**2)
                elif coord_str == "XY":
                    dist = np.sqrt(
                        (y_pred_agg[:, 0] - y_true_agg[:, 0])**2 +
                        (y_pred_agg[:, 1] - y_true_agg[:, 1])**2
                    )
                elif coord_str == "XZ":
                    dist = np.sqrt(
                        (y_pred_agg[:, 0] - y_true_agg[:, 0])**2 +
                        (y_pred_agg[:, 1] - y_true_agg[:, 1])**2
                    )
                elif coord_str == "YZ":
                    dist = np.sqrt(
                        (y_pred_agg[:, 0] - y_true_agg[:, 0])**2 +
                        (y_pred_agg[:, 1] - y_true_agg[:, 1])**2
                    )
                elif coord_str == "XYZ":
                    dist = np.sqrt(
                        (y_pred_agg[:, 0] - y_true_agg[:, 0])**2 +
                        (y_pred_agg[:, 1] - y_true_agg[:, 1])**2 +
                        (y_pred_agg[:, 2] - y_true_agg[:, 2])**2
                    )
                pass_mean_displacements.append(np.mean(dist))
            y_pred_inv = _inverse_transform(scaler, avg_preds, coord_str, **params)
            y_true_agg_mean = _aggregate_sequence_predictions(y_true_inv, row_counts, test_mode=True, **params)
            y_pred_agg_mean = _aggregate_sequence_predictions(y_pred_inv, row_counts, test_mode=True, **params)
            _assign_results(y_true_agg_mean, y_pred_agg_mean, coord_str)

        else:
            _ = model.predict(test_X[:1])
            start_time = time.time()
            y_pred_test = model.predict(test_X)
            end_time = time.time()
            total_inference_time = end_time - start_time
            avg_inference_time_per_sample = total_inference_time / test_X.shape[0]
            if verbose:
                print("*"*50 + "TCN" + "*"*50)
                print(f"\n--- Inference Time for coordinate [{coord_str}] ---")
                print(f"Total test samples: {test_X.shape[0]}")
                print(f"Total inference time  : {total_inference_time:.4f} s")
                print(f"Avg inference per sample: {avg_inference_time_per_sample:.6f} s\n")

            y_true_inv = _inverse_transform(scaler, test_y, coord_str, **params)
            y_pred_inv = _inverse_transform(scaler, y_pred_test, coord_str, **params)
            y_true_agg = _aggregate_sequence_predictions(y_true_inv, row_counts, test_mode=True, **params)
            y_pred_agg = _aggregate_sequence_predictions(y_pred_inv, row_counts, test_mode=True, **params)
            _assign_results(y_true_agg, y_pred_agg, coord_str)

        _plot_loss(history, coord_str, **params)

    _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, row_counts, test_mode=True, **params)
    metrics_dict = _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params)

    if use_mc_dropout:
        average_mean_disp = np.mean(pass_mean_displacements)
        average_std_disp  = np.std(pass_mean_displacements)
        metrics_dict["Mean_3D_Displacement"] = average_mean_disp
        metrics_dict["Std_3D_Displacement"]  = average_std_disp

    _export_metrics(metrics_dict, **params)

    return metrics_dict
