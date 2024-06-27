# main.py
from Preprocessing import Preprocess
from Training import Train
from utils.log_results import log_results, aggregate_predictions
import numpy as np

params = {
    "data_directory": "./Data/",
    "verbose": True,
    "warmup": False,
    "sequence_length": 10,
    "num_train": 7,
    "num_val": 1,
    "num_test": 2,
    "num_epochs": 5,
    "batch_size": 16,
    "sample_index": 0,
}

def train_and_evaluate(coordinate):
    params['coordinate'] = coordinate
    preprocessor = Preprocess(**params)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocessor.preprocess_data(coordinate)

    trainer = Train(**params)
    y_pred, history = trainer.train_model(X_train, X_test, X_val, y_train, y_val, coordinate)
    y_test, y_pred = trainer.test_model(y_test, y_pred, scaler, coordinate)
    
    return y_test, y_pred

if __name__ == "__main__":
    # Train and evaluate for combined X and Z coordinates
    print(f"Training model for combined X and Z coordinates...")
    XZ_test, XZ_pred = train_and_evaluate('XZ')

    # Train and evaluate for Y coordinate
    print(f"Training model for Y coordinate...")
    Y_test, Y_pred = train_and_evaluate('Y')

    # Aggregate predictions
    XZ_true, XZ_pred_aggregated = aggregate_predictions(XZ_test, XZ_pred, params['sequence_length'])
    Y_true, Y_pred_aggregated = aggregate_predictions(Y_test, Y_pred, params['sequence_length'])

    # Split aggregated XZ into X and Z
    X_true = XZ_true[:, 0]
    Z_true = XZ_true[:, 1]
    X_pred = XZ_pred_aggregated[:, 0]
    Z_pred = XZ_pred_aggregated[:, 1]

    # Log the results
    log_results(X_true, Y_true.flatten(), Z_true, X_pred, Y_pred_aggregated.flatten(), Z_pred, results_folder="./Results/", verbose=True)