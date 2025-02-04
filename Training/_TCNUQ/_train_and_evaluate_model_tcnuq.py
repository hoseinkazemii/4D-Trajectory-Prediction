# _train_and_evaluate_model_tcnuq.py
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from Preprocessing import _inverse_transform
from utils import (_aggregate_sequence_predictions,
                   _save_prediction_results, _plot_loss)
from utils._evaluate_metrics import _compute_metrics, _export_metrics

def enable_dropout(model):
    """Enable dropout layers during inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def predict_with_mc_dropout(model, data, num_samples=50):
    """
    Perform multiple forward passes with dropout enabled to obtain a distribution of predictions.
    
    Args:
        model: The trained model.
        data: A batch of data (with attributes x, edge_index, batch, node_time).
        num_samples: Number of stochastic forward passes.
    
    Returns:
        mean_pred: Mean prediction over samples.
        std_pred: Standard deviation over predictions.
    """
    # Save current training mode and set model to eval
    model_mode = model.training
    model.eval()
    enable_dropout(model)  # Enable dropout for MC sampling
    
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(data.x, data.edge_index, data.batch, data.node_time)
            preds.append(pred.unsqueeze(0))
    preds = torch.cat(preds, dim=0)  # Shape: (num_samples, batch_size, output_dim)
    mean_pred = preds.mean(dim=0)     # (batch_size, output_dim)
    std_pred = preds.std(dim=0)       # (batch_size, output_dim)
    
    # Restore original training mode
    model.train(model_mode)
    return mean_pred, std_pred

def _train_and_evaluate_model_tcnuq(split_data_dict,
                                    scalers_dict,  # e.g., {"XYZ": <scaler>}
                                    row_counts,    # for aggregation if needed
                                    **params):
    """
    Trains and evaluates the TCN model using MC Dropout for uncertainty estimation.
    Uses standard MSE loss during training.
    At test time, multiple stochastic forward passes are performed to obtain mean predictions and uncertainty.
    """
    device             = params.get('device')
    prediction_horizon = params.get('prediction_horizon')
    batch_size         = params.get('batch_size')
    num_epochs         = params.get('num_epochs')
    learning_rate      = params.get('learning_rate')
    model              = params.get('model')
    verbose            = params.get('verbose', True)

    # Prepare DataLoaders (using keys similar to your GNN setup)
    train_dataset = split_data_dict['gnn']['graphs_train']
    val_dataset   = split_data_dict['gnn']['graphs_val']
    test_dataset  = split_data_dict['gnn']['graphs_test']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_loss_history = []
    val_loss_history   = []

    # ---------------- TRAINING LOOP ----------------
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

        # ---------------- VALIDATION ----------------
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
            print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")

    # ---------------- TEST EVALUATION ----------------
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

            # Reshape to (batch_size, prediction_horizon, 3)
            pred_reshaped = pred.view(-1, prediction_horizon, 3).cpu().numpy()
            y_true_reshaped = y_true.view(-1, prediction_horizon, 3).cpu().numpy()

            all_preds.append(pred_reshaped)
            all_targets.append(y_true_reshaped)

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    if verbose:
        print(f"Final Test Loss: {avg_test_loss:.6f}")

    # Concatenate predictions and targets across mini-batches
    y_pred_test = torch.tensor([item for batch in all_preds for item in batch], dtype=torch.float)
    y_true_test = torch.tensor([item for batch in all_targets for item in batch], dtype=torch.float)
    if verbose:
        print("y_pred_test shape:", y_pred_test.shape)
        print("y_true_test shape:", y_true_test.shape)

    # ---------------- INVERSE TRANSFORM & AGGREGATE ----------------
    coord_str = "XYZ"  # assuming targets are stored as XYZ
    scaler = scalers_dict.get(coord_str, None)
    if scaler is not None:
        y_pred_inv = _inverse_transform(scaler, y_pred_test.numpy(), coord_str, **params)
        y_true_inv = _inverse_transform(scaler, y_true_test.numpy(), coord_str, **params)
    else:
        y_pred_inv = y_pred_test.numpy()
        y_true_inv = y_true_test.numpy()

    y_pred_agg = _aggregate_sequence_predictions(y_pred_inv, row_counts, **params)
    y_true_agg = _aggregate_sequence_predictions(y_true_inv, row_counts, **params)

    X_true = y_true_agg[..., 0]
    Y_true = y_true_agg[..., 1]
    Z_true = y_true_agg[..., 2]
    X_pred = y_pred_agg[..., 0]
    Y_pred = y_pred_agg[..., 1]
    Z_pred = y_pred_agg[..., 2]

    _save_prediction_results(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, row_counts, **params)
    metrics_dict = _compute_metrics(X_true, Y_true, Z_true, X_pred, Y_pred, Z_pred, **params)
    _export_metrics(metrics_dict, **params)

    # ---------------- PLOT TRAINING LOSS HISTORY ----------------
    history = {
        "history": {
            "epoch": list(range(1, num_epochs + 1)),
            "train_loss": train_loss_history,
            "val_loss": val_loss_history
        }
    }
    _plot_loss(history, coord_str, **params)

    # ---------------- PLOT PREDICTIONS WITH CONFIDENCE INTERVALS ----------------
    # For demonstration, we pick one batch from the test set and perform MC dropout.
    sample_batch = next(iter(test_loader)).to(device)
    mean_pred, std_pred = predict_with_mc_dropout(model, sample_batch, num_samples=50)
    # Select the first sample from the batch
    sample_mean = mean_pred[0].view(prediction_horizon, 3).cpu().numpy()
    sample_std = std_pred[0].view(prediction_horizon, 3).cpu().numpy()
    time_steps = np.arange(1, prediction_horizon + 1)

    # Plot the X coordinate (index 0) along with its 95% confidence interval.
    pred_x = sample_mean[:, 0]
    std_x = sample_std[:, 0]
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, pred_x, label='Predicted X', color='green')
    plt.fill_between(time_steps,
                     pred_x - 1.96 * std_x,
                     pred_x + 1.96 * std_x,
                     color='green', alpha=0.3, label='95% Confidence Interval')
    plt.xlabel('Prediction Time Step')
    plt.ylabel('X Coordinate')
    plt.title('TCNUQ (MC Dropout) Prediction with 95% Confidence Intervals\n(Sample from Test Batch)')
    plt.legend()
    plt.grid(True)
    plt.savefig('TCNUQ_MC_prediction_CI.png')
    plt.show()

    return metrics_dict
