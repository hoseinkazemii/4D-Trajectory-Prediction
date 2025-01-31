import numpy as np
from ._generate_sequences import _generate_sequences
from ._build_subset_graphs import _build_subset_graphs

def _split_data_by_scenario(scaled_arrays_list, **params):
    """
    Splits the scaled_arrays_list into train/val/test subsets.
    Additionally, if use_gnn=True, it creates graph snapshots (chain graphs)
    for each time-series window.

    scaled_arrays_list : list of dict
        Each element is a scenario_dict with keys like "XYZ", "VX", "VY", "VZ", "AX", "AY", "AZ" ...
        e.g. scenario_dict["XYZ"] => shape (T,3) for T timesteps

    Returns
    -------
    split_data_dict : dict
        If use_gnn=False:
            {
              "XYZ": {
                 "X_train": np.array(...), "y_train": np.array(...),
                 "X_val":   np.array(...), "y_val":   np.array(...),
                 "X_test":  np.array(...), "y_test":  np.array(...)
              },
              "VX": {...},
              "VY": {...},
              ...
            }
        If use_gnn=True:
            {
              "gnn": {
                 "X_train": np.array(...), "y_train": np.array(...),
                 "X_val":   np.array(...), "y_val":   np.array(...),
                 "X_test":  np.array(...), "y_test":  np.array(...),
                 "graphs_train": [...],
                 "graphs_val":   [...],
                 "graphs_test":  [...]
              }
            }
    """
    train_indices     = params.get("train_indices")
    val_indices       = params.get("val_indices")
    test_indices      = params.get("test_indices")
    coordinates       = params.get("coordinates")
    use_gnn           = params.get("use_gnn")
    use_velocity      = params.get("use_velocity")
    use_acceleration  = params.get("use_acceleration")
    verbose           = params.get("verbose", True)

    total_scenarios = len(scaled_arrays_list)
    if verbose:
        print(f"Splitting {total_scenarios} scenarios.")
        print(f"  Train indices: {train_indices}")
        print(f"  Val   indices: {val_indices}")
        print(f"  Test  indices: {test_indices}")
        if use_gnn:
            print("[GNN mode is ON] Will build chain graphs for each subset.")
            if use_velocity or use_acceleration:
                print(f"Additional Node Features => velocity={use_velocity}, acceleration={use_acceleration}")

    # Helper function: map scenario index -> subset label
    def get_subset_label(i):
        if i in train_indices:
            return "train"
        elif i in val_indices:
            return "val"
        elif i in test_indices:
            return "test"
        return None

    # Depending on use_gnn, we prepare a different data structure
    if not use_gnn:
        # coordinate-by-coordinate structure
        split_data_dict = {}
        # Prepare keys
        for coord_str in coordinates:
            split_data_dict[coord_str] = {
                "X_train": [], "y_train": [],
                "X_val": [],   "y_val":   [],
                "X_test": [],  "y_test":  [],
            }

        # For each scenario, only store sequences for the subset it belongs to
        for i, scenario_dict in enumerate(scaled_arrays_list):
            subset_label = get_subset_label(i)
            if subset_label is None:
                continue

            for coord_str in coordinates:
                if coord_str not in scenario_dict:
                    continue  # skip if missing data

                arr = scenario_dict[coord_str]  # shape (T, d)
                X_seq, y_seq = _generate_sequences(arr, **params)

                if subset_label == "train":
                    split_data_dict[coord_str]["X_train"].append(X_seq)
                    split_data_dict[coord_str]["y_train"].append(y_seq)
                elif subset_label == "val":
                    split_data_dict[coord_str]["X_val"].append(X_seq)
                    split_data_dict[coord_str]["y_val"].append(y_seq)
                elif subset_label == "test":
                    split_data_dict[coord_str]["X_test"].append(X_seq)
                    split_data_dict[coord_str]["y_test"].append(y_seq)

        # Concatenate across all scenarios for each subset
        for coord_str in coordinates:
            for subset_name in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
                if split_data_dict[coord_str][subset_name]:
                    split_data_dict[coord_str][subset_name] = np.concatenate(
                        split_data_dict[coord_str][subset_name], axis=0
                    )
                else:
                    split_data_dict[coord_str][subset_name] = np.array([])

        return split_data_dict

    else:
        # GNN approach:
        # We will combine all needed features into a single matrix: [X, Y, Z, (VX, VY, VZ)?, (AX, AY, AZ)?]
        # Then produce chain-graph snapshots from each sequence.
        # We'll store them in a single dict keyed by "gnn".
        split_data_dict = {
            "gnn": {
                "X_train": [], "y_train": [],
                "X_val":   [], "y_val":   [],
                "X_test":  [], "y_test":  [],
                # Optionally store list of graph objects or adjacency info
                "graphs_train": [],
                "graphs_val":   [],
                "graphs_test":  [],
            }
        }

        # For each scenario in scaled_arrays_list
        for i, scenario_dict in enumerate(scaled_arrays_list):
            subset_label = get_subset_label(i)
            if subset_label is None:
                continue

            # 1) Collect node features columns from scenario_dict
            #    Always add XYZ if present
            #    Then add velocity components if use_velocity=True
            #    Then add acceleration components if use_acceleration=True
            node_feature_list = []

            # If we specifically store "XYZ" as shape (T,3), add that:
            if "XYZ" in scenario_dict:
                node_feature_list.append(scenario_dict["XYZ"])  # shape (T,3)

            # If user wants velocity & scenario has them
            if use_velocity:
                for v_str in ["VX", "VY", "VZ"]:
                    if v_str in scenario_dict:
                        node_feature_list.append(scenario_dict[v_str])  # shape (T,1)

            # If user wants acceleration & scenario has them
            if use_acceleration:
                for a_str in ["AX", "AY", "AZ"]:
                    if a_str in scenario_dict:
                        node_feature_list.append(scenario_dict[a_str])  # shape (T,1)

            # Combine horizontally => shape (T, num_features)
            # e.g. if we have XYZ (3) + VX, VY, VZ (3) + AX, AY, AZ (3) => 9 columns total
            if not node_feature_list:
                # If for some reason there's no data, skip
                continue
            combined_features = np.hstack(node_feature_list)

            # 2) Generate input (X_seq) and output (y_seq) windows from the combined feature matrix
            X_seq, y_seq = _generate_sequences(combined_features, **params)

            # shapes:
            #  X_seq: (num_samples, sequence_length, num_features)
            #  y_seq: (num_samples, prediction_horizon, num_features)

            # 3) Build chain-graph adjacency (or other graph structure) for each window
            #    `_build_subset_graphs` is your custom function that, for each (X_seq[i], y_seq[i]),
            #    can create a data structure for GNN training (e.g. adjacency, PyG Data, etc.)
            #    We assume it returns a list of graph objects or adjacency matrices of the same length as X_seq.
            subset_graphs = _build_subset_graphs(X_seq, y_seq, subset_label, **params)
            # For example, each element in subset_graphs might be a dictionary or PyG `Data` object.

            # 4) Append them to the correct subset
            if subset_label == "train":
                split_data_dict["gnn"]["X_train"].append(X_seq)
                split_data_dict["gnn"]["y_train"].append(y_seq)
                split_data_dict["gnn"]["graphs_train"].extend(subset_graphs)

            elif subset_label == "val":
                split_data_dict["gnn"]["X_val"].append(X_seq)
                split_data_dict["gnn"]["y_val"].append(y_seq)
                split_data_dict["gnn"]["graphs_val"].extend(subset_graphs)

            elif subset_label == "test":
                split_data_dict["gnn"]["X_test"].append(X_seq)
                split_data_dict["gnn"]["y_test"].append(y_seq)
                split_data_dict["gnn"]["graphs_test"].extend(subset_graphs)
            
            print(f"*****************************************************************{subset_label}")

        # 5) Concatenate across all scenarios for each subset (X/Y). Graph lists are just extended.
        for subset_name in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
            if split_data_dict["gnn"][subset_name]:
                split_data_dict["gnn"][subset_name] = np.concatenate(split_data_dict["gnn"][subset_name], axis=0)
            else:
                split_data_dict["gnn"][subset_name] = np.array([])

        # For the lists of graphs, we do not need concatenation, because they are typically
        # a Python list of separate graph objects. They are already extended above.
        return split_data_dict
    