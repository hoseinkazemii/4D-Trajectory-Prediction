import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set font sizes globally
plt.rcParams.update({
    'font.size': 16,          # Default font size for all text
    'axes.titlesize': 20,     # Title font size
    'axes.labelsize': 18,     # X, Y, Z label size
    'legend.fontsize': 16,    # Legend font size
    'xtick.labelsize': 14,    # X-tick labels size
    'ytick.labelsize': 14     # Y-tick labels size
})

def plot_3d_trajectory(csv_path, **params):
    verbose = params.get("verbose")
    if verbose:
        print("Plotting the predicted trajectory vs true trajectory")

    # Load the data from CSV
    df = pd.read_csv(csv_path)

    # Extract true and predicted coordinates
    X_true = df['X_true'].values
    Y_true = df['Y_true'].values
    Z_true = df['Z_true'].values
    X_pred = df['X_predicted'].values
    Y_pred = df['Y_predicted'].values
    Z_pred = df['Z_predicted'].values

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot true trajectory
    ax.plot(X_true, Y_true, Z_true, label='True Trajectory', color='b', marker='o', linewidth=2)
    # Plot predicted trajectory
    ax.plot(X_pred, Y_pred, Z_pred, label='Predicted Trajectory', color='r', marker='x', linewidth=2)

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Save plot to a file
    plt.savefig("./Reports/TrajectoryFigures/TrueTrajectory_vs_PredictedTrajectory.png", dpi=300)
    plt.show()
