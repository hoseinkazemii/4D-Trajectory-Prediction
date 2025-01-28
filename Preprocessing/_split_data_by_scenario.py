import numpy as np
from ._generate_sequences import _generate_sequences
from ._build_subset_graphs import _build_subset_graphs


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
    train_indices     = params.get("train_indices")
    val_indices       = params.get("val_indices")
    test_indices      = params.get("test_indices")
    coordinates       = params.get("coordinates")
    use_gnn          = params.get("use_gnn")
    use_velocity     = params.get("use_velocity")
    use_acceleration = params.get("use_acceleration")
    verbose           = params.get("verbose", True)
    sequence_length = params.get("sequence_length")

    total_scenarios = len(scaled_arrays_list)
    if verbose:
        print(f"Splitting {total_scenarios} scenarios: {train_indices} for train, {val_indices} for val, {test_indices} for test.")
        if use_gnn:
            print("[GNN mode is ON] Will build chain graphs for each subset.")
            if use_velocity or use_acceleration:
                print(f"Additional Node Features => velocity={use_velocity}, acceleration={use_acceleration}")

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

    # If GNN requested, build chain-graph snippets
    if use_gnn:
        # We'll store graph data in a parallel structure: 
        # split_data_dict["graph_data"][coord_str]["train"] = list of (node_feats, edge_index, target_feats)
        graph_data_dict = {}
        for coord_str in coordinates:
            graph_data_dict[coord_str] = {
                "train": [],
                "val":   [],
                "test":  []
            }

        # Build chain adjacency once for snippet of length=sequence_length
        src_nodes = np.arange(sequence_length - 1)
        dst_nodes = np.arange(1, sequence_length)
        edge_index = np.stack([src_nodes, dst_nodes], axis=0)  # shape (2, sequence_length-1)

        # For each coordinate, build train/val/test
        for coord_str in coordinates:
            graph_data_dict[coord_str]["train"] = _build_subset_graphs(coord_str, "train", split_data_dict, edge_index, **params)
            graph_data_dict[coord_str]["val"]   = _build_subset_graphs(coord_str, "val", split_data_dict, edge_index, **params)
            graph_data_dict[coord_str]["test"]  = _build_subset_graphs(coord_str, "test", split_data_dict, edge_index, **params)

        # Attach to main dictionary
        split_data_dict["graph_data"] = graph_data_dict

        if verbose:
            for coord_str in coordinates:
                n_tr = len(graph_data_dict[coord_str]["train"])
                n_va = len(graph_data_dict[coord_str]["val"])
                n_te = len(graph_data_dict[coord_str]["test"])
                print(f"[{coord_str}] GNN chain-graphs => train={n_tr}, val={n_va}, test={n_te}")

    # Return either just arrays (if use_gnn=False) or arrays + graph_data (if use_gnn=True)
    return split_data_dict
