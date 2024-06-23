from Preprocessing import Preprocess
from Training import Train

params = {
    "data_directory": "./Data/",
    "verbose": True,
    "warmup": False,
    "sequence_length": 10,
    "train_data_split": 0.8,
    "num_epochs": 50,
    "batch_size": 32,
    "validation_split": 0.2,
    "sample_index": 0,
}

def main():
    # Preprocess the data
    preprocessor = Preprocess(**params)
    X_train, X_test, y_train, y_test, scaler = preprocessor.preprocess_data()

    # Train the model
    trainer = Train(**params)
    y_pred = trainer.train_model(X_train, X_test, y_train)
    trainer.test_model(y_test, y_pred, scaler)

if __name__ == "__main__":
    main()