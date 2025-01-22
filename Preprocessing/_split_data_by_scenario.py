import numpy as np
from ._generate_sequences import _generate_sequences

def _split_data_by_scenario(scaled_arrays_list, **params):
    """
    scaled_arrays_list: list of scenario_dicts, each scenario_dict => { coord_str: scaled_array }
      e.g. scaled_arrays_list[i]["X"] => shape (rows_i,1)

    Instead of fixed integer counts for train/val/test, now we allow lists:
      - train_indices: List of indices assigned to training
      - val_indices: List of indices assigned to validation
      - test_indices: List of indices assigned to testing

    Returns: split_data_dict => e.g. { "X": { "X_train": <>, "y_train": <>, ... }, "Y": {...}, ... }
    """

    train_indices = params.get("train_indices")
    val_indices   = params.get("val_indices")
    test_indices  = params.get("test_indices")
    coordinates       = params.get("coordinates")
    verbose          = params.get("verbose", True)

    total_scenarios = len(scaled_arrays_list)
    
    if verbose:
        print(f"Splitting {total_scenarios} scenarios: {train_indices} for train, {val_indices} for val, {test_indices} for test.")

    # Prepare empty structure
    split_data_dict = {}
    for coordinate_str in coordinates:
        split_data_dict[coordinate_str] = {
            "X_train": [], "y_train": [],
            "X_val": [],   "y_val": [],
            "X_test": [],  "y_test": []
        }

    # Helper function to determine the subset
    def get_subset(i):
        if i in train_indices:
            return "train"
        elif i in val_indices:
            return "val"
        elif i in test_indices:
            return "test"

    # Loop over each scenario & coordinate
    for i, scenario_dict in enumerate(scaled_arrays_list):
        subset_label = get_subset(i)
        if subset_label is None:
            continue  # Ignore this scenario if it's not part of any subset

        for coordinate_str in coordinates:
            arr = scenario_dict[coordinate_str]  # shape (rows_i, dim)
            X_seq, y_seq = _generate_sequences(arr, **params)

            if subset_label == "train":
                split_data_dict[coordinate_str]["X_train"].append(X_seq)
                split_data_dict[coordinate_str]["y_train"].append(y_seq)
            elif subset_label == "val":
                split_data_dict[coordinate_str]["X_val"].append(X_seq)
                split_data_dict[coordinate_str]["y_val"].append(y_seq)
            elif subset_label == "test":
                split_data_dict[coordinate_str]["X_test"].append(X_seq)
                split_data_dict[coordinate_str]["y_test"].append(y_seq)

    # Concatenate within each subset
    for coordinate_str in coordinates:
        for subset_name in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
            if split_data_dict[coordinate_str][subset_name]:
                split_data_dict[coordinate_str][subset_name] = np.concatenate(split_data_dict[coordinate_str][subset_name], axis=0)
            else:
                split_data_dict[coordinate_str][subset_name] = np.array([])  # Ensure an empty array if no data

    return split_data_dict



# import numpy as np
# from ._generate_sequences import _generate_sequences

# def _split_data_by_scenario(scaled_arrays_list, **params):
#     """
#     scaled_arrays_list: list of scenario_dicts, each scenario_dict => { coord_str: scaled_array }
#       e.g. scaled_arrays_list[i]["X"] => shape (rows_i,1)

#     We'll assign:
#       - first num_train scenarios => train
#       - next num_val scenarios => val
#       - last num_test scenarios => test
#     Then generate sequences for each scenario independently, 
#     and finally concatenate them within each subset.

#     Returns: split_data_dict => e.g. { "X": { "X_train": <>, "y_train": <>, ... }, "Y": {...}, ... }
#     """
#     num_train_files = params.get("num_train")
#     num_val_files   = params.get("num_val")
#     num_test_files  = params.get("num_test")
#     coordinates     = params.get("coordinates")
#     verbose         = params.get("verbose", True)

#     total_scenarios = len(scaled_arrays_list)
#     if verbose:
#         print(f"Splitting {total_scenarios} scenarios: {num_train_files} for train, {num_val_files} for val, {num_test_files} for test.")

#     # Prepare empty structure
#     split_data_dict = {}
#     for coordinate_str in coordinates:
#         split_data_dict[coordinate_str] = {
#             "X_train": [], "y_train": [],
#             "X_val": [],   "y_val": [],
#             "X_test": [],  "y_test": []
#         }

#     # Helper to get subset label for scenario i
#     def get_subset(i):
#         if i < num_train_files:
#             return "train"
#         elif i < num_train_files + num_val_files:
#             return "val"
#         else:
#             return "test"

#     # Loop over each scenario & coordinate
#     for i, scenario_dict in enumerate(scaled_arrays_list):
#         subset_label = get_subset(i)
#         for coordinate_str in coordinates:
#             arr = scenario_dict[coordinate_str]  # shape (rows_i, dim)
#             X_seq, y_seq = _generate_sequences(arr, **params)

#             if subset_label == "train":
#                 split_data_dict[coordinate_str]["X_train"].append(X_seq)
#                 split_data_dict[coordinate_str]["y_train"].append(y_seq)
#             elif subset_label == "val":
#                 split_data_dict[coordinate_str]["X_val"].append(X_seq)
#                 split_data_dict[coordinate_str]["y_val"].append(y_seq)
#             else:  # test
#                 split_data_dict[coordinate_str]["X_test"].append(X_seq)
#                 split_data_dict[coordinate_str]["y_test"].append(y_seq)

#     # Concatenate within each subset
#     for coordinate_str in coordinates:
#         for subset_name in ["X_train","y_train","X_val","y_val","X_test","y_test"]:
#             if len(split_data_dict[coordinate_str][subset_name]) > 0:
#                 split_data_dict[coordinate_str][subset_name] = np.concatenate(split_data_dict[coordinate_str][subset_name], axis=0)
#             else:
#                 # If no data, set an empty array
#                 split_data_dict[coordinate_str][subset_name] = np.array([])

#     return split_data_dict