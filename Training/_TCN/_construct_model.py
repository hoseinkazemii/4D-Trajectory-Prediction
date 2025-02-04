# _TCN.py
import torch
import torch.nn as nn
import torch.nn.functional as F

###########################
#  Utility: Chomp1d Module
###########################
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Remove the last chomp_size elements in the time dimension.
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

#################################
#  TemporalBlock (TCN Building Block)
#################################
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        # If input and output channel numbers differ, use 1x1 conv for the residual.
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

#################################
#  TCN Model for Graph Data
#################################
class TCNModel(nn.Module):
    """
    TCNModel:
      1) Gathers node features from each graph into a sequence based on node_time.
      2) Passes the sequence through a stack of TemporalBlocks.
      3) Takes the last time-step’s representation and applies a linear layer to output
         (prediction_horizon * 3) coordinates.
    """
    def __init__(self, in_channels, hidden_dim, prediction_horizon, num_levels=2, kernel_size=3, dropout=0.5):
        super(TCNModel, self).__init__()
        self.prediction_horizon = prediction_horizon
        self.out_dim = prediction_horizon * 3

        layers = []
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else hidden_dim
            dilation = 2 ** i
            layers.append(
                TemporalBlock(in_ch, hidden_dim, kernel_size, stride=1, dilation=dilation, dropout=dropout)
            )
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, x, edge_index, batch, node_time):
        """
        x:           (num_nodes, in_channels) node features
        edge_index:  (ignored) – provided for API compatibility
        batch:       (num_nodes,) indicating which graph each node belongs to
        node_time:   (num_nodes,) local time index of each node in [0, sequence_length-1]
        """
        # 1) Gather the node features into (batch_size, sequence_length, in_channels)
        x_seq = self._gather_embeddings_by_time(x, batch, node_time)
        # 2) Transpose to (batch_size, in_channels, sequence_length) for 1D convolutions
        x_seq = x_seq.transpose(1, 2)
        # 3) Pass through the TCN stack
        y = self.tcn(x_seq)
        # 4) Take the last time step (causal – last element corresponds to the “present”) and predict output
        out = self.fc(y[:, :, -1])
        return out

    def _gather_embeddings_by_time(self, x, batch, node_time):
        """
        Rearranges the flat node embeddings x (from all graphs) into a tensor of shape:
           (num_graphs_in_batch, sequence_length, in_channels)
        """
        hidden_dim = x.size(1)
        num_graphs = batch.max().item() + 1
        seq_len = node_time.max().item() + 1
        # Initialize an empty tensor on the same device as x.
        x_seq = x.new_zeros((num_graphs, seq_len, hidden_dim))
        # For each node i, place x[i] in the proper (graph, time) slot.
        for i in range(x.size(0)):
            g = batch[i].item()
            t = node_time[i].item()
            x_seq[g, t] = x[i]
        return x_seq

#################################
#  Model Constructor
#################################
def _construct_model_tcn(**params):
    device = params.get('device')
    in_channels = params.get('in_channels')
    hidden_dim = params.get('hidden_dim')
    prediction_horizon = params.get('prediction_horizon')
    num_levels = params.get('num_levels', 2)
    kernel_size = params.get('kernel_size', 3)
    dropout = params.get('dropout', 0.5)
    verbose = params.get('verbose', True)

    if verbose:
        print(f"Constructing TCNModel with in_channels={in_channels}, hidden_dim={hidden_dim}, "
              f"prediction_horizon={prediction_horizon}, num_levels={num_levels}, "
              f"kernel_size={kernel_size}, dropout={dropout}")

    model = TCNModel(in_channels, hidden_dim, prediction_horizon,
                     num_levels=num_levels, kernel_size=kernel_size, dropout=dropout)
    model = model.to(device)
    return model
