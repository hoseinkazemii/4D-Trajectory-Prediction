import torch
from torch_geometric.data import Data

def _build_subset_graphs(X_seq, y_seq, subset_label, **params):
    """
    Build chain-graph snapshots with multi-hop forward edges.
    
    For each node i in the sequence, we create directed edges to i+1, i+2, ..., i+max_hop
    (as long as those indices are within the sequence length).

    Parameters
    ----------
    X_seq : np.ndarray of shape (num_samples, sequence_length, num_features)
        The input windows for a single subset (e.g. 'train').
    y_seq : np.ndarray of shape (num_samples, prediction_horizon, num_features)
        The target windows for the same subset.
    subset_label : str
        Indicates which subset these sequences belong to ('train', 'val', 'test').
    **params : dict
        max_hop : int (optional)
            Maximum forward hop to connect. Defaults to 1 (which is a simple chain).
            If max_hop=2, each node connects to i+1 and i+2, if possible.
        Additional parameters if needed.

    Returns
    -------
    graphs : list of torch_geometric.data.Data
        A list of Data objects (one per sample window).
    """

    max_hop = params.get("max_hop", 1)  # default = 1 => immediate chain

    graphs = []
    num_samples = X_seq.shape[0]

    for i in range(num_samples):
        # Convert X_seq[i] and y_seq[i] to tensors
        node_features = torch.tensor(X_seq[i], dtype=torch.float)  # shape (sequence_length, num_features)
        label_tensor  = torch.tensor(y_seq[i], dtype=torch.float)  # shape (prediction_horizon, num_features)

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
            x=node_features,    # Node feature matrix
            edge_index=edge_index,
            y=label_tensor      # Optionally store the label
        )

        # If you want to store subset_label or other metadata:
        data.subset_label = subset_label
        graphs.append(data)

    return graphs
