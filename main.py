import torch
from Preprocessing import Preprocess
from Training import *
from utils import plot_3d_trajectory
import numpy as np

num_scenarios = 8
num_trials_per_scenario = 29
excluded_index = 173 # 202  # The missing participant's file index
training_scenarios = [0,1,2,4,5,6,7]
test_scenario = 3


train_indices = [
    i for scenario in training_scenarios
    for i in range(scenario * num_trials_per_scenario, (scenario + 1) * num_trials_per_scenario)
    if i % num_trials_per_scenario != 0 and i != excluded_index  # Exclude first trial for validation & missing file
]
val_indices = [
    scenario * num_trials_per_scenario for scenario in range(num_scenarios)
    if scenario != test_scenario and scenario * num_trials_per_scenario != excluded_index
]
test_indices = [
    i for i in range(test_scenario * num_trials_per_scenario, (test_scenario + 1) * num_trials_per_scenario)
    if i != excluded_index  # Exclude missing file
]


common_params = {
    "test_scenario": test_scenario, # The scenario index for testing (0-7)
    "top_k": 7, # The number of similar scenarios to use for training
    "num_trials_per_scenario": 29,
    "excluded_index": 173,
    "data_directory": "./Data/UpdatedData/RandomPositions/",
    "verbose": True,
    "warmup": False, 
    "signal_based_extraction": True, 
    "time_interval": 0.5,
    "noise_std": 0.0,
    "sequence_length": 10, # The length of the input sequences (e.g., 10 time steps)
    "sequence_step": 1, # The distance between consecutive coordinates to generate sequences
    "test_stride_mode": "prediction_horizon",  # Choose "prediction_horizon" or "total_window"
    "prediction_horizon": 6, # The number of future time steps we want to predict
    "train_indices": train_indices,
    "val_indices": val_indices,
    "test_indices": test_indices,
    "num_epochs": 20,
    "learning_rate" : 0.001,
    "decay_steps" : 1000,
    "decay_rate" : 0.9,
    "l2_reg_factor": 0.0005,
    "batch_size": 32,
    "run_eagerly": True, # Set to True for visualizing attention weights
    "num_sequences_for_attn": 100,  # Number of sequences to visualize attention weights for
    "sample_index": 0,
    "coord_to_indices" : { # A helper mapping from coordinate string to the appropriate column indices
        "X":   [0],
        "Y":   [1],
        "Z":   [2],
        "XY":  [0, 1],
        "XZ":  [0, 2],
        "YZ":  [1, 2],
        "XYZ": [0, 1, 2]
    },
    "coord_to_dim" : { # A helper mapping from coordinate string to the appropriate dimension
        "X":   1,
        "Y":   1,
        "Z":   1,
        "XY":  2,
        "XZ":  2,
        "YZ":  2,
        "XYZ": 3
    },
    "coordinates": ["XYZ"], # custom coordinates to train on
    "use_gnn": False, # Set to True to use GNN; and False to use other models (i.e., Seq2Seq, TCN, etc.)
    "use_velocity": False,
    "use_acceleration": False,
    "max_hop": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "in_channels": 3, # 3 for XYZ, 3 for VXVYVZ, 3 for AXAYAZ
    "use_mc_dropout": False, # Set to True to use MC Dropout
    "mc_dropout_passes": 50, # Number of passes for MC Dropout
    "scheduled_sampling_prob": 0.5, # Probability of using the predicted value instead of the true value during Teacher Forcing
    "use_init_final_positions": False, # Set to True to use initial and final positions as additional features



}

run_specific_params = {
    "model_name": "Seq2SeqTemporalAttention", # "GNN" (GATTemporal), "TCN", "Seq2SeqTemporalAttention", "Seq2SeqMultiHeadAttention" (Seq2SeqGlobalAttention), 
                        # "ConvLSTM" , "Seq2SeqLocalAttention"
}

params = {**common_params, **run_specific_params}

def main():
    preprocessor = Preprocess(**params)
    split_data_dict, scalers_dict, row_counts = preprocessor.preprocess_data()

    trainer = Seq2SeqTemporalAttention(**params)
    trainer.construct_model()
    trainer.run(split_data_dict, scalers_dict, row_counts)


if __name__ == "__main__":

    main()
