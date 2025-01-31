from ._compute_node_features import _compute_node_features


# Build snippet-graphs for each subset
def _build_subset_graphs(coord_str, subset_name, split_data_dict, edge_index, **params):
    """
    subset_name in ["train","val","test"]
    We'll read X_{subset}, y_{subset} from split_data_dict[coord_str].
    For each sample, build (node_feats, edge_index, target_feats).
    """
    X_data = split_data_dict[coord_str]["X_" + subset_name]  # shape (N, sequence_length, D)
    y_data = split_data_dict[coord_str]["y_" + subset_name]  # shape (N, prediction_horizon, D)

    n_samples = X_data.shape[0]
    snippet_graphs = []
    for i in range(n_samples):
        positions_2d = X_data[i]   # shape (sequence_length, D)
        target_2d    = y_data[i]   # shape (prediction_horizon, D)

        # node features: possibly including velocity & acceleration
        # print(f"n_samples: {n_samples}")
        # print(f"positions_2d: {positions_2d}")
        # raise ValueError
        node_features = _compute_node_features(positions_2d, **params)  
        # node_features shape => (sequence_length, out_dim)

        # We store (node_features, edge_index, target_2d)
        snippet_graphs.append((node_features, edge_index, target_2d))

    return snippet_graphs