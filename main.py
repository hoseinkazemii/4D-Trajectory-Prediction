from Preprocessing import Preprocess
from Training import *
from utils import plot_3d_trajectory


common_params = {
    "data_directory": "./Data/",
    "verbose": True,
    "warmup": False,
    "sequence_length": 10, # The length of the input sequences (e.g., 10 time steps)
    "sequence_step": 1, # The distance between consecutive coordinates to generate sequences
    "prediction_horizon": 3, # The number of future time steps we want to predict
    "num_train": 7,
    "num_val": 1,
    "num_test": 2,
    "num_epochs": 50,
    "batch_size": 16,
    "sample_index": 0,
    "coordinates": ["Y", "XZ"], # "Y" or "XZ"
    "coord_to_indices" : { # A helper mapping from coordinate string to the appropriate column indices
        "X":   [0],
        "Y":   [1],
        "Z":   [2],
        "XY":  [0, 1],
        "XZ":  [0, 2],
        "YZ":  [1, 2],
        "XYZ": [0, 1, 2]
    },



}

run_specific_params = {
    "model_name": "Seq2SeqMultiHeadAttention",
}

params = {**common_params, **run_specific_params}

def main():
    # Step 1: Preprocess and Train models
    preprocessor = Preprocess(**params)
    split_data_dict, scalers_dict = preprocessor.preprocess_data()

    trainer = Seq2SeqMultiHeadAttention(**params)
    trainer.construct_model()
    trainer.run(split_data_dict, scalers_dict)


    # Step 2: Plot "predicted trajectory" vs "true trajectory" (CHANGE the datetime in csv_path)
    # plot_3d_trajectory(csv_path="./Reports/Seq2SeqMultiHeadAttention/202501021337/Results.csv", **params)



if __name__ == "__main__":

    main()