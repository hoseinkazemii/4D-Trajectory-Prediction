import os
import matplotlib.pyplot as plt
import pandas as pd

def _plot_loss(history, coord_str, **params):
    verbose = params.get('verbose')
    report_directory = params.get("report_directory")
    model_name = params.get("model_name")
    use_gnn = params.get("use_gnn")
    coordinates = params.get("coordinates")

    plots_path = os.path.join(report_directory, f'loss_plot_{coordinates}_{coord_str}_{model_name}.png')
    loss_file_path = os.path.join(report_directory, f"loss_per_epoch_{coord_str}_{model_name}.csv")

    if verbose:
        print(f"Plotting loss for {coordinates} model...")
        print(f"Loss per epoch saved for {coord_str}")

    if use_gnn:
        num_epochs = len(history["history"]["train_loss"])
        loss_data = {
            "epoch": list(range(1, num_epochs + 1)),
            "train_loss": history["history"]["train_loss"],
            "val_loss": history["history"]["val_loss"]
        }
    else:
        num_epochs = len(history.history["loss"])
        loss_data = {
            "epoch": list(range(1, num_epochs + 1)),
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"]
        }

    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv(loss_file_path, index=False)

    plt.figure(figsize=(10, 6))

    epochs = list(range(1, num_epochs + 1))

    plt.plot(epochs, loss_data["train_loss"], label='Train Loss')
    plt.plot(epochs, loss_data["val_loss"], label='Validation Loss')

    plt.title(f'Model Loss for {coordinates}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    tick_values = list(range(5, num_epochs + 1, 5))
    if num_epochs % 5 != 0 and tick_values[-1] != num_epochs:
        tick_values.append(num_epochs)

    plt.xticks(tick_values)

    plt.savefig(plots_path)
    plt.close()
