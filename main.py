from Preprocessing import Preprocess
from Training import *
from utils import plot_3d_trajectory


common_params = {
    "data_directory": "./Data/UpdatedData/RandomPositions/",
    "verbose": True,
    "warmup": False,
    "signal_based_extraction": True,
    "time_interval": 1.0,
    "sequence_length": 10, # The length of the input sequences (e.g., 10 time steps)
    "sequence_step": 1, # The distance between consecutive coordinates to generate sequences
    "prediction_horizon": 3, # The number of future time steps we want to predict
    "train_indices": list(range(0,2)),
    "val_indices": list(range(2,3)),
    "test_indices": list(range(3,4)),
    "num_epochs": 50,
    "learning_rate" : 0.001,
    "decay_steps" : 1000,
    "decay_rate" : 0.9,
    "batch_size": 32,
    "run_eagerly":False,
    "sample_index": 0,
    "coordinates": ["XYZ"], # custom coordinates
    "coord_to_indices" : { # A helper mapping from coordinate string to the appropriate column indices
        "X":   [0],
        "Y":   [1],
        "Z":   [2],
        "XY":  [0, 1],
        "XZ":  [0, 2],
        "YZ":  [1, 2],
        "XYZ": [0, 1, 2]
    },
    "coord_to_dim" : {
        "X":   1,
        "Y":   1,
        "Z":   1,
        "XY":  2,
        "XZ":  2,
        "YZ":  2,
        "XYZ": 3
    },
    "use_gnn": True,
    "use_velocity": True,
    "use_acceleration": True,



}

run_specific_params = {
    "model_name": "Seq2SeqTemporalAttention",
}

params = {**common_params, **run_specific_params}

def main():
    # Step 1: Preprocess and Train models
    preprocessor = Preprocess(**params)
    split_data_dict, scalers_dict, row_counts = preprocessor.preprocess_data()
    trainer = Seq2SeqTemporalAttention(**params)
    trainer.construct_model()
    trainer.run(split_data_dict, scalers_dict, row_counts)


    # Step 2: Plot "predicted trajectory" vs "true trajectory" (CHANGE the datetime in csv_path)
    # plot_3d_trajectory(csv_path="./Reports/Seq2SeqTemporalAttention/202501211047/Results_TestSet_1.csv", **params)



if __name__ == "__main__":

    main()