import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.nn import global_mean_pool


class GATTemporalModel(nn.Module):
    """
    GNN:
      1) Uses GATConv layers with residual connections + BatchNorm
      2) Gathers node embeddings by their time index
      3) Aggregates them with an LSTM readout
      4) Finally outputs (prediction_horizon * 3) coords

    Requires that each Data object has:
      - data.x: shape (num_nodes, in_channels)
      - data.edge_index
      - data.batch
      - data.node_time: shape (num_nodes,) with local time indices from 0..(sequence_length-1)
        (We can also store these in data.x)
    """
    def __init__(self, in_channels, hidden_dim, prediction_horizon, heads=4):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.out_dim = prediction_horizon * 3

        # GATConv layers
        # 1st GAT: in_channels -> hidden_dim, multi-head
        self.gat1 = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
        self.bn1 = BatchNorm(hidden_dim * heads)
        # self.proj_in1 = nn.Linear(in_features=in_channels, out_features=hidden_dim * heads, bias=False)

        # 2nd GAT: hidden_dim * heads -> hidden_dim
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=True)
        self.bn2 = BatchNorm(hidden_dim)
        # self.proj_in2 = nn.Linear(in_features=hidden_dim * heads, out_features=hidden_dim, bias=False)

        # LSTM for temporal readout
        # We'll map hidden_dim -> hidden_dim in the LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Final linear for (prediction_horizon * 3)
        self.fc = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, x, edge_index, batch, node_time):
        """
        x         : FloatTensor, shape (num_nodes_total, in_channels)
        edge_index: LongTensor, shape (2, E)
        batch     : LongTensor, which graph each node belongs to
        node_time : LongTensor, shape (num_nodes_total,), time index of each node in [0..seq_len-1]
                    or we can also store it in data.x

        Return:  shape (batch_size, prediction_horizon*3)
        """
        # ---- GAT Layer #1 with Residual ----
        # x_res = x
        x = self.gat1(x, edge_index)      # => shape (num_nodes, hidden_dim*heads)
        x = self.bn1(x)
        x = F.elu(x)

        # Project x_res to match dimension
        # x_res = self.proj_in1(x_res)       # => (num_nodes, hidden_dim*heads)

        # x = x + x_res  # same shape => (num_nodes, hidden_dim*heads)
        x = F.dropout(x, 0.5, training=self.training)

        # ---- GAT Layer #2 with Residual ----
        # x_res = x
        x = self.gat2(x, edge_index)               # (num_nodes_total, hidden_dim)
        x = self.bn2(x)
        x = F.elu(x)

        # x_res = self.proj_in2(x_res)  # project x_res to hidden_dim

        # x = x + x_res  # residual
        x = F.dropout(x, p=0.5, training=self.training)

        # ---- LSTM Readout by Time ----
        # 1) Group node embeddings by (graph_id, time_index)
        #    We'll form a (batch_size, sequence_length, hidden_dim) for each graph.
        #    Each graph has 'sequence_length' nodes, with time_index in [0..seq_len-1].

        # gather_embeddings is a helper function that rearranges x -> (bsize, seq_len, hidden_dim)
        x_seq = self._gather_embeddings_by_time(x, batch, node_time)
        # x_seq: shape (num_graphs_in_batch, seq_len, hidden_dim)

        # 2) Pass through LSTM
        #    We only want the final hidden state as the readout for each graph
        #    (Though we can also do an attention across timesteps, etc.)
        _, (h_n, _) = self.lstm(x_seq)  # h_n shape => (num_layers, batch_size, hidden_dim)
        # We'll take the last layer's hidden state:
        graph_emb = h_n[-1]  # shape => (batch_size, hidden_dim)

        # 3) Final linear => (batch_size, prediction_horizon*3)
        out = self.fc(graph_emb)
        return out

    def _gather_embeddings_by_time(self, x, batch, node_time):
        """
        Reorder node embeddings 'x' into shape (num_graphs_in_batch, seq_len, hidden_dim)
        where seq_len = the max time_index + 1.

        Each graph in the batch has exactly the same sequence_length of nodes.
        """
        hidden_dim = x.size(1)
        num_graphs = batch.max().item() + 1
        seq_len = node_time.max().item() + 1

        # Create an empty tensor for (num_graphs, seq_len, hidden_dim)
        x_seq = x.new_zeros((num_graphs, seq_len, hidden_dim))

        # We can fill it by simple indexing:
        # for each node i, graph_id = batch[i], time_id = node_time[i]
        # x_seq[graph_id, time_id] = x[i]
        for i in range(x.size(0)):
            g = batch[i].item()
            t = node_time[i].item()
            x_seq[g, t] = x[i]

        return x_seq


def _construct_model(**params):
    device = params.get('device')
    in_channels = params.get('in_channels')
    hidden_dim = params.get('hidden_dim')
    prediction_horizon = params.get('prediction_horizon')
    num_heads = params.get('num_heads')
    verbose = params.get('verbose', True)

    if verbose:
        print(f"Constructing GATTemporalModel with in_channels={in_channels}, hidden_dim={hidden_dim}, "
              f"prediction_horizon={prediction_horizon}, heads={num_heads}")

    model = GATTemporalModel(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        prediction_horizon=prediction_horizon,
        heads=num_heads
    ).to(device)

    return model




############################################################################################################
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool
# from torch_geometric.nn import GATConv, GraphConv, SAGEConv, BatchNorm

# class GATTrajectoryPredictor(nn.Module):
#     def __init__(self, in_channels, hidden_dim, prediction_horizon, heads=4):
#         super().__init__()
#         self.prediction_horizon = prediction_horizon
#         out_dim = prediction_horizon * 3

#         # GATConv layers
#         self.gat1 = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
#         self.bn1 = BatchNorm(hidden_dim * heads)
#         self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=True)
#         self.bn2 = BatchNorm(hidden_dim)

#         self.fc = nn.Linear(hidden_dim, out_dim)

#     def forward(self, x, edge_index, batch):
#         x = self.gat1(x, edge_index)
#         x = self.bn1(x)
#         x = F.elu(x)
#         x = F.dropout(x, 0.2, training=self.training)

#         x = self.gat2(x, edge_index)
#         x = self.bn2(x)
#         x = F.elu(x)
#         x = F.dropout(x, 0.2, training=self.training)

#         x = global_mean_pool(x, batch)
#         return self.fc(x)
    
# def _construct_model(**params):
#     """
#     Constructs a single GNN model for XYZ forecasting.
#     """
#     device = params.get('device')
#     verbose = params.get('verbose', True)
#     in_channels        = params.get('in_channels')           # e.g. 9 if [XYZ, VX, VY, VZ, AX, AY, AZ]
#     hidden_dim = params.get('hidden_dim')
#     prediction_horizon = params.get('prediction_horizon')

#     if verbose:
#         print(f"Constructing GNN model with in_channels={in_channels}, hidden_dim={hidden_dim}, prediction_horizon={prediction_horizon}")

#     # Build Model
#     model = GATTrajectoryPredictor(
#         in_channels=in_channels,
#         hidden_dim=hidden_dim,
#         prediction_horizon=prediction_horizon
#     ).to(device)

#     return model