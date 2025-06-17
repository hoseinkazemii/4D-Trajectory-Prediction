from ._load_data import _load_data
from ._df_to_array import _dfs_to_array
from ._scale_data import _scale_data
from ._split_data_by_scenario import _split_data_by_scenario
from ._visualize_aggregated_distributions import _visualize_aggregated_distributions
from ._scenario_distance_matrix import scenario_distance_matrix
from ._get_similar_scenarios import _get_similar_scenarios

import numpy as np


class Preprocess():
    def __init__(self, **params):
        self.params = params

        self.df_list = None
        self.row_counts = None
        self.arrays_list = None
        self.scaled_arrays_list = None
        self.scalers_dict = None
        self.split_data_dict = None
        self.graph_data_dict = None

    def preprocess_data(self):
        self.df_list, self.row_counts = _load_data(**self.params)
        self.arrays_list = _dfs_to_array(self.df_list, **self.params)

        distance_matrix = np.load("scenario_distance_matrix.npy")
        test_scenario = self.params.get("test_scenario")
        top_k = self.params.get("top_k")
        num_trials_per_scenario = self.params.get("num_trials_per_scenario")
        excluded_index = self.params.get("excluded_index")

        similar_indices = _get_similar_scenarios(distance_matrix, test_scenario, top_k=top_k)
        
        train_indices = [
            i for s in similar_indices
            for i in range(s * num_trials_per_scenario, (s + 1) * num_trials_per_scenario)
            if i % num_trials_per_scenario != 0 and i != excluded_index
        ]
        val_indices = [
            s * num_trials_per_scenario for s in similar_indices
            if s != test_scenario and s * num_trials_per_scenario != excluded_index
        ]
        test_indices = [
            i for i in range(test_scenario * num_trials_per_scenario, (test_scenario + 1) * num_trials_per_scenario)
            if i != excluded_index
        ]

        self.params.update({
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices
        })

        print(f"Test scenario: {test_scenario}")
        print(f"Top-{top_k} similar scenarios for training: {similar_indices}")
        print(f"Train indices: {train_indices}")
        print(f"Validation indices: {val_indices}")
        print(f"Test indices: {test_indices}")

        self.scaled_arrays_list, self.scalers_dict = _scale_data(self.arrays_list, **self.params)
        self.split_data_dict = _split_data_by_scenario(self.scaled_arrays_list, **self.params)

        return self.split_data_dict, self.scalers_dict, self.row_counts
