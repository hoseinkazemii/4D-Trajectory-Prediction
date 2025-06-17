import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

class GATTemporalModel(nn.Module):
    def __init__(self, in_channels, prediction_horizon):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.out_dim = prediction_horizon * 3

        self.gat1 = GATConv(in_channels, out_channels=64, heads=2, concat=True)
        self.bn1 = BatchNorm(in_channels=64*2)
        self.proj_in1 = nn.Linear(in_features=in_channels, out_features=64*2, bias=False)

        self.gat2 = GATConv(in_channels=64*2, out_channels=64, heads=1, concat=True)
        self.bn2 = BatchNorm(in_channels=64)
        self.proj_in2 = nn.Linear(in_features=64*2, out_features=32, bias=False)

        self.gat3 = GATConv(in_channels=128, out_channels=64, heads=1, concat=True)
        self.bn3 = BatchNorm(in_channels=64)
        self.proj_in3 = nn.Linear(in_features=128, out_features=64, bias=False)

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, dropout=0.2, num_layers=2, batch_first=True)

        self.fc = nn.Linear(in_features=64, out_features=self.out_dim)

    def forward(self, x, edge_index, batch, node_time):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.gat2(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x_seq = self._gather_embeddings_by_time(x, batch, node_time)

        _, (h_n, _) = self.lstm(x_seq)
        graph_emb = h_n[-1]

        out = self.fc(graph_emb)
        return out

    def _gather_embeddings_by_time(self, x, batch, node_time):
        hidden_dim = x.size(1)
        num_graphs = batch.max().item() + 1
        seq_len = node_time.max().item() + 1

        x_seq = x.new_zeros((num_graphs, seq_len, hidden_dim))

        for i in range(x.size(0)):
            g = batch[i].item()
            t = node_time[i].item()
            x_seq[g, t] = x[i]

        return x_seq