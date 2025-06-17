import time
import numpy as np
from Preprocessing import _inverse_transform
from utils import _aggregate_sequence_predictions, _save_prediction_results, _plot_loss
from utils._evaluate_metrics import _compute_metrics, _export_metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from ._attention_layer import TemporalEncDecAttention

def _train_and_evaluate_model(split_data_dict, scalers_dict, row_counts, **params):
    coordinates = params.get("coordinates")
    verbose = params.get('verbose', True)
    num_epochs = params.get('num_epochs')
    batch_size = params.get('batch_size')
    models_dict = params.get("models_dict")
    model_name = params.get("model_name")
    run_eagerly = params.get("run_eagerly")
    report_directory = params.get("report_directory")
    num_sequences_for_attn = params.get("num_sequences_for_attn")

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

        history = model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=1 if verbose else 0
        )

        _ = model.predict(test_X[:1])

        start_time = time.time()
        y_pred_test = model.predict(test_X)
        end_time = time.time()

        total_inference_time = end_time - start_time
        avg_inference_time_per_sample = total_inference_time / test_X.shape[0]

        if verbose:
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

    if run_eagerly:
        attention_layer = None
        for layer in model.layers:
            if isinstance(layer, TemporalEncDecAttention):
                attention_layer = layer
                break

        num_sequences_for_attn = min(num_sequences_for_attn, test_X.shape[0])
        cumulative_attention_weights = None

        for i in range(num_sequences_for_attn):
            input_sample = test_X[i:i+1]
            _ = model.predict(input_sample)

            attention_weights = attention_layer.get_attention_weights().numpy()

            if cumulative_attention_weights is None:
                cumulative_attention_weights = attention_weights
            else:
                cumulative_attention_weights += attention_weights

        avg_attention_weights = cumulative_attention_weights / num_sequences_for_attn

        np.save(
            f'{report_directory}/avg_attention_weights_{coord_str}.npy',
            {
                'weights': avg_attention_weights,
                'seq_length': params['sequence_length'],
                'pred_horizon': params['prediction_horizon'],
                'num_heads': attention_layer.num_heads
            },
            allow_pickle=True
        )

    _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, row_counts, test_mode=True, **params)
    metrics_dict = _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params)
    _export_metrics(metrics_dict, **params)

    return metrics_dict
