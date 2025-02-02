import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------------------------------------------
# GNN Architecture
# ---------------------------------------------
class GNNModel(nn.Module):
    """
    A simple example GNN:
      - Two GCNConv layers with ReLU,
      - Global mean pooling to produce a single graph-level embedding,
      - Final linear output of size (prediction_horizon * 3).

    Each input graph is a chain (or multi-hop) of 'sequence_length' nodes.
    Each node has 'in_channels' features (e.g. [X,Y,Z, VX, ...]).
    """
    def __init__(self, in_channels, hidden_dim, prediction_horizon, **params):        
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon
        # We assume we want to predict X,Y,Z => out_dim = horizon * 3
        out_dim = prediction_horizon * 3

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        # GCN #1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # GCN #2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Global mean pooling => one embedding per graph
        x = global_mean_pool(x, batch)

        # Final linear => shape (batch_size, prediction_horizon*3)
        out = self.fc(x)
        return out


def _construct_model(**params):
    """
    Constructs a single GNN model for XYZ forecasting.
    """
    device = params.get('device')
    verbose = params.get('verbose', True)
    in_channels        = params.get('in_channels')           # e.g. 9 if [XYZ, VX, VY, VZ, AX, AY, AZ]
    hidden_dim = params.get('hidden_dim')
    prediction_horizon = params.get('prediction_horizon')

    if verbose:
        print(f"Constructing GNN model with in_channels={in_channels}, hidden_dim={hidden_dim}, prediction_horizon={prediction_horizon}")

    # Build Model
    model = GNNModel(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        prediction_horizon=prediction_horizon
    ).to(device)

    return model