# main.py
from Preprocessing import Preprocess
from Training import Train

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
    trainer.test_model(y_test, y_pred, scaler, coordinate)

if __name__ == "__main__":
    
    print(f"Training model for combined X and Z coordinates...")
    train_and_evaluate(['X', 'Z'])

    # print(f"Training model for Y coordinate...")
    # train_and_evaluate('Y')