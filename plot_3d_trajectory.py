import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_trajectory(csv_path, output_path):
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
    ax.plot(X_true, Y_true, Z_true, label='True Trajectory', color='b', marker='o')

    # Plot predicted trajectory
    ax.plot(X_pred, Y_pred, Z_pred, label='Predicted Trajectory', color='r', marker='x')

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('True vs Predicted 3D Trajectory')
    ax.legend()

    # Save plot to a file
    plt.savefig(output_path, dpi=300)
    plt.close()

# Example usage
csv_path = "./Results/results.csv"
output_path = "./Results/trajectory_plot.png"
plot_3d_trajectory(csv_path, output_path)