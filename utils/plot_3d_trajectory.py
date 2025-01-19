import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Set font sizes globally
plt.rcParams.update({
    'font.size': 16,          
    'axes.titlesize': 20,     
    'axes.labelsize': 18,     
    'legend.fontsize': 16,    
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14     
})

def plot_3d_trajectory(csv_path, **params):
    model_name = params.get("model_name", "Model")
    coordinates = params.get("coordinates", "Coords")
    verbose = params.get("verbose", True)
    show_errors = params.get("show_errors")
    error_threshold = params.get("error_threshold")

    if verbose:
        print(f"Plotting the predicted trajectory vs true trajectory for {model_name}")

    filename = os.path.basename(csv_path)
    filename = os.path.splitext(filename)[0]

    # Load the data
    df = pd.read_csv(csv_path)

    # Extract true and predicted coordinates
    X_true = df['X_true'].values
    Y_true = df['Z_true'].values  # Unity's Z -> Cartesian Y
    Z_true = df['Y_true'].values  # Unity's Y -> Cartesian Z
    X_pred = df['X_predicted'].values
    Y_pred = df['Z_predicted'].values
    Z_pred = df['Y_predicted'].values

    # Compute Euclidean distance error
    point_errors = np.sqrt((X_true - X_pred) ** 2 + (Y_true - Y_pred) ** 2 + (Z_true - Z_pred) ** 2)

    # Identify high-error points (above threshold)
    high_error_indices = point_errors > error_threshold

    # --------- 3D Trajectory Visualization --------- #
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot true trajectory (blue)
    ax.plot(X_true, Y_true, Z_true, label='True Trajectory', color='b', marker='o', linewidth=2)

    # Plot predicted trajectory (red)
    ax.plot(X_pred, Y_pred, Z_pred, label='Predicted Trajectory', color='r', marker='x', linewidth=2)

    # Highlight points where error exceeds threshold (red)
    ax.scatter(X_pred[high_error_indices], Y_pred[high_error_indices], Z_pred[high_error_indices], 
               color='red', s=50, label=f'Error > {error_threshold}m')

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f"True vs Predicted Trajectory - {model_name}")

    # Save plot
    os.makedirs("./Reports/TrajectoryFigures", exist_ok=True)
    plt.savefig(f"./Reports/TrajectoryFigures/Trajectory_{model_name}_{coordinates}_{filename}.png", dpi=300)
    plt.show()

    # --------- Heatmap Visualization --------- #
    if show_errors:
        if verbose:
            print("Performing point-wise error analysis...")

        fig, ax = plt.subplots(figsize=(12, 6))
        sc = ax.scatter(X_true, Y_true, c=point_errors, cmap='coolwarm', marker='o', edgecolors='k')
        plt.colorbar(sc, label="Error (m)")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(f"Heatmap of Point-wise Prediction Errors - {model_name}")

        # Save heatmap
        os.makedirs("./Reports/ErrorAnalysis", exist_ok=True)
        plt.savefig(f"./Reports/ErrorAnalysis/Heatmap_{model_name}_{coordinates}_{filename}.png", dpi=300)
        plt.show()

        # Print error summary
        mean_error = np.mean(point_errors)
        max_error = np.max(point_errors)
        if verbose:
            print(f"Mean Prediction Error: {mean_error:.4f} meters")
            print(f"Max Prediction Error: {max_error:.4f} meters")
            print(f"Total points with error > {error_threshold}m: {np.sum(high_error_indices)}")

