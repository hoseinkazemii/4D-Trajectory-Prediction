import pandas as pd
import matplotlib.pyplot as plt

# Set font sizes globally
plt.rcParams.update({
    'font.size': 16,          # Default font size for all text
    'axes.titlesize': 20,     # Title font size
    'axes.labelsize': 18,     # X and Y label size
    'legend.fontsize': 16,    # Legend font size
    'xtick.labelsize': 14,    # X-tick labels size
    'ytick.labelsize': 14     # Y-tick labels size
})

# List of file names and corresponding labels
files = [
    ("loss_per_epoch_XYZ_Seq2SeqLuongAttention.csv", "Luong Attention"),
    ("loss_per_epoch_XYZ_Seq2SeqMultiHeadAttention.csv", "Multi-Head Attention"),
    ("loss_per_epoch_XYZ_Seq2SeqTemporalAttention.csv", "Temporal Attention"),
]

# Colors for each model
colors = ['red', 'blue', 'green']

# Initialize dictionaries to store data
train_loss_data = {}
val_loss_data = {}

# Load data from CSV files
for file, label in files:
    data = pd.read_csv(file)
    train_loss_data[label] = data['train_loss']
    val_loss_data[label] = data['val_loss']

# Plot training loss per epoch
plt.figure(figsize=(10, 6))
for (label, train_loss), color in zip(train_loss_data.items(), colors):
    plt.plot(train_loss, label=label, color=color, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.grid()
plt.savefig("TrainingLossPerEpoch_AllModels.png", format="png", dpi=300)
plt.show()

# Plot validation loss per epoch
plt.figure(figsize=(10, 6))
for (label, val_loss), color in zip(val_loss_data.items(), colors):
    plt.plot(val_loss, label=label, color=color, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid()
plt.savefig("ValidationLossPerEpoch_AllModels.png", format="png", dpi=300)
plt.show()

