import torch
from torch_geometric.data import Data

def _build_subset_graphs(X_seq, y_seq, subset_label, **params):
    """
    Build chain-graph snapshots with multi-hop forward edges, 
    and attach a time index (node_time) for each node.

    Parameters
    ----------
    X_seq : np.ndarray of shape (num_samples, sequence_length, num_features)
        The input windows for a single subset (e.g. 'train').
    y_seq : np.ndarray of shape (num_samples, prediction_horizon, num_features)
        The target windows for the same subset.
    subset_label : str
        Indicates which subset these sequences belong to ('train', 'val', 'test').
    max_hop : int (optional, via params)
        If max_hop=2, each node connects to i+1 and i+2, if possible.

    Returns
    -------
    graphs : list of torch_geometric.data.Data
        A list of Data objects (one per sample window).
    """
    max_hop = params.get("max_hop", 1)

    graphs = []
    num_samples = X_seq.shape[0]

    for i in range(num_samples):
        # Convert X_seq[i] and y_seq[i] to torch tensors
        node_features = torch.tensor(X_seq[i], dtype=torch.float)  # shape (sequence_length, num_features)
        label_tensor  = torch.tensor(y_seq[i], dtype=torch.float)  # shape (prediction_horizon, <target_dim>)

        sequence_length = node_features.size(0)

        # Build adjacency with up to 'max_hop' forward edges from each node i
        src = []
        dst = []
        for node_idx in range(sequence_length):
            for h in range(1, max_hop + 1):
                neighbor = node_idx + h
                if neighbor < sequence_length:
                    src.append(node_idx)
                    dst.append(neighbor)

        edge_index = torch.tensor([src, dst], dtype=torch.long)  # shape (2, num_edges)

        # Create the PyG Data object
        data = Data(
            x=node_features,       # Node feature matrix of shape (sequence_length, in_features)
            edge_index=edge_index, # Directed edges with max_hop
            y=label_tensor         # The label: shape (prediction_horizon, <dims>)
        )

        # --------- ATTACH THE NODE_TIME TENSOR HERE ---------
        # If each node in X_seq[i] corresponds to time=0..(sequence_length-1),
        # we can store that in data.node_time
        data.node_time = torch.arange(sequence_length, dtype=torch.long)

        # Optionally store subset label
        data.subset_label = subset_label

        graphs.append(data)

    return graphs

