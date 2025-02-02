import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import torch.optim as optim

from Preprocessing import _inverse_transform
from utils import (_aggregate_sequence_predictions,
                    _save_prediction_results, _plot_loss)
from utils._evaluate_metrics import _compute_metrics, _export_metrics

# ---------------------------------------------
# Training & Evaluation
# ---------------------------------------------
def _train_and_evaluate_model(split_data_dict,
                                  scalers_dict,     # e.g. {"XYZ": <some scaler>}, if you want inverse_transform
                                  row_counts,       # for aggregator
                                  **params):
    """
    Trains a single GNN for XYZ forecasting, evaluates on val/test,
    does inverse transform & aggregation, computes metrics, plots loss, etc.

    This is the GNN analog to the Seq2Seq _train_and_evaluate_model.
    """
    # Hyperparameters
    device             = params.get('device')
    prediction_horizon = params.get('prediction_horizon')
    batch_size         = params.get('batch_size')
    num_epochs         = params.get('num_epochs')
    learning_rate      = params.get('learning_rate')
    model = params.get('model')
    verbose            = params.get('verbose', True)

    # Prepare PyG DataLoaders
    train_dataset = split_data_dict['gnn']['graphs_train']
    val_dataset   = split_data_dict['gnn']['graphs_val']
    test_dataset  = split_data_dict['gnn']['graphs_test']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # For plotting losses
    train_loss_history = []
    val_loss_history   = []

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            pred = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            # pred shape => (batch_size, prediction_horizon*3)

            # Flatten targets => shape (batch_size, prediction_horizon*3)
            y_true = batch_data.y.view(-1, prediction_horizon * 3).to(device)

            loss = criterion(pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Multiply by batch_data.num_graphs because 'loss.item()' is averaged over the batch
            total_train_loss += loss.item() * batch_data.num_graphs

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        if len(val_dataset) > 0:
            with torch.no_grad():
                for batch_data in val_loader:
                    batch_data = batch_data.to(device)
                    pred = model(batch_data.x, batch_data.edge_index, batch_data.batch)
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

    # ---------------- TEST EVALUATION ----------------
    model.eval()
    total_test_loss = 0.0

    # We'll collect predictions & targets for final analysis
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            pred = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            y_true = batch_data.y.view(-1, prediction_horizon * 3)

            loss = criterion(pred, y_true)
            total_test_loss += loss.item() * batch_data.num_graphs

            # Reshape to (batch_size, prediction_horizon, 3)
            pred_reshaped = pred.view(-1, prediction_horizon, 3).cpu().numpy()
            y_true_reshaped = y_true.view(-1, prediction_horizon, 3).cpu().numpy()

            all_preds.append(pred_reshaped)
            all_targets.append(y_true_reshaped)

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    if verbose:
        print(f"Final Test Loss: {avg_test_loss:.6f}")

    # Concatenate everything across mini-batches
    y_pred_test = torch.tensor([], dtype=torch.float)
    y_true_test = torch.tensor([], dtype=torch.float)

    if len(all_preds) > 0:
        y_pred_test = torch.tensor(
            [item for batch in all_preds for item in batch],  # flatten list-of-batches
            dtype=torch.float
        )
        y_true_test = torch.tensor(
            [item for batch in all_targets for item in batch],
            dtype=torch.float
        )
    # => shape (N_test_samples, prediction_horizon, 3)

    # (Optional) Print shape for debugging
    if verbose:
        print("y_pred_test shape:", y_pred_test.shape)
        print("y_true_test shape:", y_true_test.shape)

    # ---------------- INVERSE TRANSFORM & AGGREGATE ----------------
    # If you have a single scaler for "XYZ" you can do:
    coord_str = "XYZ"  # We assume we have 3 coords in the GNN
    scaler = scalers_dict.get(coord_str, None)  # or handle differently if you have multiple scalers
    if scaler is not None:
        # shape => (N, horizon, 3)
        y_pred_inv = _inverse_transform(scaler, y_pred_test.numpy(), coord_str, **params)
        y_true_inv = _inverse_transform(scaler, y_true_test.numpy(), coord_str, **params)
    else:
        # If no scaler, just take them as they are
        y_pred_inv = y_pred_test.numpy()
        y_true_inv = y_true_test.numpy()

    # If needed, you can aggregate sliding-window predictions. 
    # For example, if each row in 'row_counts' corresponds to a separate segment, 
    # you can do:
    y_pred_agg = _aggregate_sequence_predictions(y_pred_inv, row_counts, **params)
    y_true_agg = _aggregate_sequence_predictions(y_true_inv, row_counts, **params)

    # ---------------- SAVE & METRICS ----------------
    # This next part parallels your _assign_results step, but we know we have XYZ
    # So let's just treat them as separate arrays:
    X_true = y_true_agg[..., 0]
    Y_true = y_true_agg[..., 1]
    Z_true = y_true_agg[..., 2]
    X_pred = y_pred_agg[..., 0]
    Y_pred = y_pred_agg[..., 1]
    Z_pred = y_pred_agg[..., 2]

    _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, row_counts, **params)
    metrics_dict = _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params)
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
