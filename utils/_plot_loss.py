import matplotlib.pyplot as plt

def _plot_loss(history, coordinate, **params):
    verbose = params.get('verbose')
    if verbose:
        print(f"Plotting loss for {coordinate} model...")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss for {coordinate}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_plot_{coordinate}.png')
    plt.close()