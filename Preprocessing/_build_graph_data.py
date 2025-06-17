import numpy as np

def combine_features_for_sample(idx_array_dict, **params):
    coordinates = params.get("coordinates")
    feat_list = []
    for coord in coordinates:
        feat_list.append(idx_array_dict[coord].squeeze(-1))
    node_features = np.stack(feat_list, axis=1)
    return node_features


def build_subset_graphs(subset_name, split_data_dict, edge_index, **params):
    coordinates = params.get("coordinates")

    subset_arrays_input = {}
    subset_arrays_target = {}

    for coord in coordinates:
        subset_arrays_input[coord]  = split_data_dict[coord]["X_" + subset_name]
        subset_arrays_target[coord] = split_data_dict[coord]["y_" + subset_name]

    num_samples = subset_arrays_input[coordinates[0]].shape[0]

    subset_graphs = []
    for i in range(num_samples):
        sample_input_dict = {}
        for coord in coordinates:
            sample_input_dict[coord] = subset_arrays_input[coord][i]

        node_features = combine_features_for_sample(sample_input_dict, **params)

        sample_target_dict = {}
        for coord in coordinates:
            sample_target_dict[coord] = subset_arrays_target[coord][i]

        feat_list_target = []
        for coord in coordinates:
            feat_list_target.append(sample_target_dict[coord].squeeze(-1)) 
        target_features = np.stack(feat_list_target, axis=1)

        graph_tuple = (node_features, edge_index, target_features)
        subset_graphs.append(graph_tuple)

    return subset_graphs


def _build_graph_data(split_data_dict, **params):
    sequence_length = params.get("sequence_length")
    verbose = params.get("verbose", True)

    src_nodes = np.arange(sequence_length - 1)
    dst_nodes = np.arange(1, sequence_length)
    edge_index = np.stack([src_nodes, dst_nodes], axis=0)

    graph_data_dict = {
        "train": [],
        "val":   [],
        "test":  []
    }
    graph_data_dict["train"] = build_subset_graphs("train", split_data_dict, edge_index, **params)
    graph_data_dict["val"]   = build_subset_graphs("val", split_data_dict, edge_index, **params)
    graph_data_dict["test"]  = build_subset_graphs("test", split_data_dict, edge_index, **params)

    if verbose:
        print(f"Built GNN chain-graphs: train={len(graph_data_dict['train'])}, "
              f"val={len(graph_data_dict['val'])}, test={len(graph_data_dict['test'])}")

    return graph_data_dict
