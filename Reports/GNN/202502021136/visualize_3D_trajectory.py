import pandas as pd
import matplotlib.pyplot as plt

for i in range(1, 6):
    # Load the data from the CSV file
    csv_file = f"Results_TestSet_{i}.csv"  # Replace with the path to your CSV file
    data = pd.read_csv(csv_file)

    # Extract the true and predicted trajectories
    X_true, Y_true, Z_true = data["X_true"], data["Y_true"], data["Z_true"]

    X_pred, Y_pred, Z_pred = data["X_predicted"], data["Y_predicted"], data["Z_predicted"]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the true trajectory
    ax.plot(X_true, Y_true, Z_true, label="True Trajectory", linewidth=2, linestyle="-", marker="o")

    # Plot the predicted trajectory
    ax.plot(X_pred, Y_pred, Z_pred, label="Predicted Trajectory", linewidth=2, linestyle="--", marker="x")

    # Set labels and title
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("True vs Predicted Trajectory")
    ax.legend()
    plt.savefig(f"trajectory_{csv_file}.png")
    # Show the plot
    plt.show()
