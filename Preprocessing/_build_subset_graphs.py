import torch
from torch_geometric.data import Data

def _build_subset_graphs(X_seq, y_seq, subset_label, **params):
    max_hop = params.get("max_hop")

    graphs = []
    num_samples = X_seq.shape[0]

    for i in range(num_samples):
        node_features = torch.tensor(X_seq[i], dtype=torch.float)
        label_tensor  = torch.tensor(y_seq[i], dtype=torch.float)

        sequence_length = node_features.size(0)

        src = []
        dst = []
        for node_idx in range(sequence_length):
            for h in range(1, max_hop + 1):
                neighbor = node_idx + h
                if neighbor < sequence_length:
                    src.append(node_idx)
                    dst.append(neighbor)

        edge_index = torch.tensor([src, dst], dtype=torch.long)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=label_tensor
        )

        data.node_time = torch.arange(sequence_length, dtype=torch.long)

        data.subset_label = subset_label

        graphs.append(data)

    return graphs
