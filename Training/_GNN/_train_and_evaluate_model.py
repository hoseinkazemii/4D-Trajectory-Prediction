import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR

from Preprocessing import _inverse_transform
from utils import (_aggregate_sequence_predictions,
                    _save_prediction_results, _plot_loss)
from utils._evaluate_metrics import _compute_metrics, _export_metrics

def _train_and_evaluate_model(split_data_dict,
                                  scalers_dict,
                                  row_counts,
                                  **params):
    device             = params.get('device')
    prediction_horizon = params.get('prediction_horizon')
    batch_size         = params.get('batch_size')
    num_epochs         = params.get('num_epochs')
    learning_rate      = params.get('learning_rate')
    model = params.get('model')
    verbose            = params.get('verbose', True)
    decay_rate         = params.get('decay_rate')

    train_dataset = split_data_dict['gnn']['graphs_train']
    val_dataset   = split_data_dict['gnn']['graphs_val']
    test_dataset  = split_data_dict['gnn']['graphs_test']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=decay_rate)

    criterion = nn.HuberLoss(delta=1.0)

    train_loss_history = []
    val_loss_history   = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            pred = model(batch_data.x, batch_data.edge_index, batch_data.batch, batch_data.node_time)

            y_true = batch_data.y.view(-1, prediction_horizon * 3).to(device)

            loss = criterion(pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_data.num_graphs

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        if len(val_dataset) > 0:
            with torch.no_grad():
                for batch_data in val_loader:
                    batch_data = batch_data.to(device)
                    pred = model(batch_data.x, batch_data.edge_index, batch_data.batch, batch_data.node_time)
                    y_true = batch_data.y.view(-1, prediction_horizon * 3)
                    loss = criterion(pred, y_true)
                    total_val_loss += loss.item() * batch_data.num_graphs

            avg_val_loss = total_val_loss / len(val_loader.dataset)
        else:
            avg_val_loss = float('nan')
        val_loss_history.append(avg_val_loss)

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}]  "
                  f"Train Loss: {avg_train_loss:.6f}  "
                  f"Val Loss: {avg_val_loss:.6f}")

    scheduler.step()

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Learning Rate: {scheduler.get_last_lr()[0]}")

    model.eval()
    total_test_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            pred = model(batch_data.x, batch_data.edge_index, batch_data.batch, batch_data.node_time)
            y_true = batch_data.y.view(-1, prediction_horizon * 3)

            loss = criterion(pred, y_true)
            total_test_loss += loss.item() * batch_data.num_graphs

            pred_reshaped = pred.view(-1, prediction_horizon, 3).cpu().numpy()
            y_true_reshaped = y_true.view(-1, prediction_horizon, 3).cpu().numpy()

            all_preds.append(pred_reshaped)
            all_targets.append(y_true_reshaped)

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    if verbose:
        print(f"Final Test Loss: {avg_test_loss:.6f}")

    y_pred_test = torch.tensor([], dtype=torch.float)
    y_true_test = torch.tensor([], dtype=torch.float)

    if len(all_preds) > 0:
        y_pred_test = torch.tensor(
            [item for batch in all_preds for item in batch],
            dtype=torch.float
        )
        y_true_test = torch.tensor(
            [item for batch in all_targets for item in batch],
            dtype=torch.float
        )

    coord_str = "XYZ"
    scaler = scalers_dict.get(coord_str, None)
    if scaler is not None:
        y_pred_inv = _inverse_transform(scaler, y_pred_test.numpy(), coord_str, **params)
        y_true_inv = _inverse_transform(scaler, y_true_test.numpy(), coord_str, **params)
    else:
        y_pred_inv = y_pred_test.numpy()
        y_true_inv = y_true_test.numpy()

    y_pred_agg = _aggregate_sequence_predictions(y_pred_inv, row_counts, test_mode=True, **params)
    y_true_agg = _aggregate_sequence_predictions(y_true_inv, row_counts, test_mode=True, **params)

    X_true = y_true_agg[..., 0]
    Y_true = y_true_agg[..., 1]
    Z_true = y_true_agg[..., 2]
    X_pred = y_pred_agg[..., 0]
    Y_pred = y_pred_agg[..., 1]
    Z_pred = y_pred_agg[..., 2]

    _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, row_counts, test_mode=True, **params)
    metrics_dict = _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, test_mode=True, **params)
    _export_metrics(metrics_dict, **params)

    history = {
        "history": {
            "epoch": list(range(1, num_epochs + 1)),
            "train_loss": train_loss_history,
            "val_loss": val_loss_history
        }
    }

    _plot_loss(history, coord_str, **params)

    return metrics_dict
