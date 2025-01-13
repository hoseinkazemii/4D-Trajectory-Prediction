import os
import matplotlib.pyplot as plt
import pandas as pd

def _plot_loss(history, coordinate, coord_str, **params):
    verbose = params.get('verbose')
    report_directory = params.get("report_directory")
    model_name = params.get("model_name")

    plots_path = os.path.join(report_directory, f'loss_plot_{coordinate}_{model_name}.png')
    loss_file_path = os.path.join(report_directory, f"loss_per_epoch_{coord_str}_{model_name}.csv")

    if verbose:
        print(f"Plotting loss for {coordinate} model...")
    if verbose:
        print(f"Loss per epoch saved for {coord_str}")

    # Save loss per epoch to a CSV file
    loss_data = {
        "epoch": list(range(1, len(history.history["loss"]) + 1)),
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"]
    }

    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv(loss_file_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss for {coordinate}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_path)
    plt.close()