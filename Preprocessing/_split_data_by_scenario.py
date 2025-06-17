import numpy as np
from ._generate_sequences import _generate_sequences
from ._build_subset_graphs import _build_subset_graphs

def _split_data_by_scenario(scaled_arrays_list, **params):
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
    
    def get_subset_label(i):
        if i in train_indices:
            return "train"
        elif i in val_indices:
            return "val"
        elif i in test_indices:
            return "test"
        return None

    if not use_gnn:
        split_data_dict = {}
        for coord_str in coordinates:
            split_data_dict[coord_str] = {
                "X_train": [], "y_train": [],
                "X_val": [],   "y_val":   [],
                "X_test": [],  "y_test":  [],
            }

        for i, scenario_dict in enumerate(scaled_arrays_list):
            subset_label = get_subset_label(i)
            if subset_label is None:
                continue

            for coord_str in coordinates:
                if coord_str not in scenario_dict:
                    continue

                arr = scenario_dict[coord_str]
                if subset_label == "test":
                    X_seq, y_seq = _generate_sequences(arr, test_mode=True, **params)
                else:
                    X_seq, y_seq = _generate_sequences(arr, test_mode=False, **params)

                if subset_label == "train":
                    split_data_dict[coord_str]["X_train"].append(X_seq)
                    split_data_dict[coord_str]["y_train"].append(y_seq)
                elif subset_label == "val":
                    split_data_dict[coord_str]["X_val"].append(X_seq)
                    split_data_dict[coord_str]["y_val"].append(y_seq)
                elif subset_label == "test":
                    split_data_dict[coord_str]["X_test"].append(X_seq)
                    split_data_dict[coord_str]["y_test"].append(y_seq)

        for coord_str in coordinates:
            for subset_name in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
                if split_data_dict[coord_str][subset_name]:
                    split_data_dict[coord_str][subset_name] = np.concatenate(
                        split_data_dict[coord_str][subset_name], axis=0
                    )
                else:
                    split_data_dict[coord_str][subset_name] = np.array([])

        return split_data_dict

    elif use_gnn:
        split_data_dict = {
            "gnn": {
                "graphs_train": [],
                "graphs_val":   [],
                "graphs_test":  [],
            }
        }

        for i, scenario_dict in enumerate(scaled_arrays_list):
            subset_label = get_subset_label(i)
            if subset_label is None:
                continue

            node_feature_list = []

            if "XYZ" in scenario_dict:
                node_feature_list.append(scenario_dict["XYZ"])

            if use_velocity:
                for v_str in ["VX", "VY", "VZ"]:
                    if v_str in scenario_dict:
                        node_feature_list.append(scenario_dict[v_str])

            if use_acceleration:
                for a_str in ["AX", "AY", "AZ"]:
                    if a_str in scenario_dict:
                        node_feature_list.append(scenario_dict[a_str])

            if not node_feature_list:
                continue
            combined_features = np.hstack(node_feature_list)

            if subset_label == "test":
                X_seq, y_seq = _generate_sequences(combined_features, test_mode=True, **params)
            else:
                X_seq, y_seq = _generate_sequences(combined_features, test_mode=False, **params)

            subset_graphs = _build_subset_graphs(X_seq, y_seq, subset_label, **params)

            if subset_label == "train":
                split_data_dict["gnn"]["graphs_train"].extend(subset_graphs)
            elif subset_label == "val":
                split_data_dict["gnn"]["graphs_val"].extend(subset_graphs)
            elif subset_label == "test":
                split_data_dict["gnn"]["graphs_test"].extend(subset_graphs)

        return split_data_dict
