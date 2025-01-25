import numpy as np

def _build_graph_data(split_data_dict, **params):
    """
    Build chain-graph data for each snippet of the time series.
    We assume 'split_data_dict' has structure like:
      split_data_dict["X"]["X_train"], ["X_val"], ["X_test"]
      split_data_dict["Y"]["X_train"], etc.
      ...
    Where these are numpy arrays of shape (num_samples, sequence_length, 1) or similar.
    
    We want to produce something like:
      graph_data = {
         "train": [ (node_features_1, edge_index_1, y_target_1), 
                    (node_features_2, edge_index_2, y_target_2), ... ],
         "val":   [ ... ],
         "test":  [ ... ],
      }
    
    This skeleton just shows how you might do it for X, Y, Z combined. 
    Extend or modify for velocity, acceleration, etc.
    """
    # You can decide how to gather coordinates. Suppose "coordinates" = ["X", "Y", "Z"] in params:
    coordinates = params.get("coordinates")
    sequence_length = params.get("sequence_length")
    verbose = params.get("verbose", True)

    # We will build the chain adjacency once for a snippet of length = sequence_length
    # edge_index typically in shape (2, E) for PyTorch Geometric, for instance:
    # chain edges: (0->1, 1->2, ..., sequence_length-2->sequence_length-1)
    # we'll do them in an undirected manner or directed, depending on the GNN.
    # Example for directed edges from t to t+1:
    src_nodes = np.arange(sequence_length - 1)
    dst_nodes = np.arange(1, sequence_length)
    edge_index = np.stack([src_nodes, dst_nodes], axis=0)  # shape (2, sequence_length - 1)

    # We'll store final results here:
    graph_data_dict = {
        "train": [],
        "val":   [],
        "test":  []
    }

    # We'll define a helper function to combine features for each coordinate & time step
    # e.g. for a single sample we might get node_features shape (sequence_length, 3) for X,Y,Z
    def combine_features_for_sample(idx_array_dict):
        # idx_array_dict is e.g. { "X": (sequence_length,1), "Y": (sequence_length,1), "Z": (sequence_length,1) }
        # We'll just horizontally stack them: shape -> (sequence_length, 3)
        feat_list = []
        for coord in coordinates:
            feat_list.append(idx_array_dict[coord].squeeze(-1))  # shape => (sequence_length,)
        # shape => (sequence_length, len(coordinates))
        node_features = np.stack(feat_list, axis=1)
        return node_features

    # We'll define a small routine to gather input/target from the split_data_dict for a given subset
    def build_subset_graphs(subset_name):
        # For each coordinate, we have X_{subset_name} with shape (N, sequence_length, 1)
        # Also y_{subset_name} with shape (N, prediction_horizon, 1)
        # We want to combine X, Y, Z into single node_features. Then store the target for training.

        # Let's gather them in a structure, e.g.:
        #   subset_arrays = { "X": X_sub, "Y": Y_sub, "Z": Z_sub }
        #   each shape => (N, sequence_length, 1) for the input side
        subset_arrays_input = {}
        subset_arrays_target = {}

        for coord in coordinates:
            # e.g. split_data_dict[coord]["X_train"] is shape => (N, sequence_length, 1)
            #     split_data_dict[coord]["y_train"] is shape => (N, prediction_horizon, 1)
            subset_arrays_input[coord]  = split_data_dict[coord]["X_" + subset_name]
            subset_arrays_target[coord] = split_data_dict[coord]["y_" + subset_name]

        # We assume they are the same length (N)
        if not subset_arrays_input[coordinates[0]].size:
            # Means no data in this subset, return empty
            return []

        N = subset_arrays_input[coordinates[0]].shape[0]  # number of samples

        subset_graphs = []
        for i in range(N):
            # Build the node features from the i-th sample across X, Y, Z
            sample_input_dict = {}
            for coord in coordinates:
                sample_input_dict[coord] = subset_arrays_input[coord][i]  # shape => (sequence_length, 1)

            node_features = combine_features_for_sample(sample_input_dict)  # shape (sequence_length, 3) if coords = [X, Y, Z]

            # For the target, you might want to combine them or keep them separate
            # Let's keep it simple and just store them as a dictionary or a single array
            # shape => (prediction_horizon, 3)
            sample_target_dict = {}
            for coord in coordinates:
                sample_target_dict[coord] = subset_arrays_target[coord][i]  # shape => (prediction_horizon, 1)

            # Combine them horizontally => (prediction_horizon, len(coordinates)) if that suits your GNN/MLP approach
            # Or just keep them separate as a dict. We'll combine for a single array:
            feat_list_target = []
            for coord in coordinates:
                feat_list_target.append(sample_target_dict[coord].squeeze(-1)) 
            # shape => (prediction_horizon, len(coordinates))
            target_features = np.stack(feat_list_target, axis=1)

            # Now we have:
            #   node_features: shape => (sequence_length, len(coordinates))
            #   edge_index: shape => (2, sequence_length-1)
            #   target_features: shape => (prediction_horizon, len(coordinates))
            # Store them as a tuple or a dictionary
            graph_tuple = (node_features, edge_index, target_features)
            subset_graphs.append(graph_tuple)

        return subset_graphs

    graph_data_dict["train"] = build_subset_graphs("train")
    graph_data_dict["val"]   = build_subset_graphs("val")
    graph_data_dict["test"]  = build_subset_graphs("test")

    if verbose:
        print(f"Built GNN chain-graphs: train={len(graph_data_dict['train'])}, "
              f"val={len(graph_data_dict['val'])}, test={len(graph_data_dict['test'])}")

    return graph_data_dict




# Suppose you have time[t], X[t], Y[t], Z[t]
# velocity_x[t] = (X[t+1] - X[t]) / (time[t+1] - time[t])
# acceleration_x[t] = velocity_x[t+1] - velocity_x[t] / dt
# etc.



# graph_data_dict = {
#   "train": [(node_feats, edge_index, target_feats), ...],
#   "val":   [...],
#   "test":  [...]
# }
