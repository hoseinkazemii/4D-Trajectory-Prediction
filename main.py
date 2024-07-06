from Preprocessing import Preprocess
from Training import Seq2SeqWithSelfAttention
from utils import plot_3d_trajectory


params = {
    "data_directory": "./Data/",
    "verbose": True,
    "warmup": False,
    "model_name": "Seq2SeqWithSelfAttention",
    "sequence_length": 10, # The length of the input sequences (e.g., 10 time steps)
    "prediction_horizon": 3, # The number of future time steps we want to predict
    "num_train": 7,
    "num_val": 1,
    "num_test": 2,
    "num_epochs": 5,
    "batch_size": 16,
    "sample_index": 0,
    "coordinates": ["Y", "XZ"], # "Y" or "XZ"
}

def main():
    # Step 1: Preprocess and Train models on Y and XZ coordinates
    preprocessor = Preprocess(**params)
    X_train_Y_coordinate, X_val_Y_coordinate, X_test_Y_coordinate, y_train_Y_coordinate, y_val_Y_coordinate, y_test_Y_coordinate, \
    X_train_XZ_coordinate, X_val_XZ_coordinate, X_test_XZ_coordinate, y_train_XZ_coordinate, y_val_XZ_coordinate, y_test_XZ_coordinate, \
    Y_scaler, XZ_scaler = preprocessor.preprocess_data()
    
    trainer = Seq2SeqWithSelfAttention(**params)
    trainer._construct_model()
    trainer.run(X_train_Y_coordinate, X_val_Y_coordinate, X_test_Y_coordinate, y_train_Y_coordinate, y_val_Y_coordinate, y_test_Y_coordinate, \
                X_train_XZ_coordinate, X_val_XZ_coordinate, X_test_XZ_coordinate, y_train_XZ_coordinate, y_val_XZ_coordinate, y_test_XZ_coordinate, \
                Y_scaler, XZ_scaler)
    
    # Step 2: Plot "predicted trajectory" vs "true trajectory" (CHANGE the datetime in csv_path)
    # plot_3d_trajectory(csv_path="./Reports/Seq2SeqWithSelfAttention/202407030211/Results.csv", **params)



if __name__ == "__main__":

    main()