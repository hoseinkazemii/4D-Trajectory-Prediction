import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV file
csv_file = 'LoadData_BaselineScenario_20250113165352.csv'
data = pd.read_csv(csv_file)
data = data[data["LoadingStarted"] == 1]

# Ensure the columns 'X', 'Y', and 'Z' exist in the dataset
if all(col in data.columns for col in ['X', 'Y', 'Z']):
    # Adjust axes to match Unity's convention
    x = data['X']  # Unity's X -> Cartesian X
    y = data['Z']  # Unity's Z (depth) -> Cartesian Y
    z = data['Y']  # Unity's Y (height) -> Cartesian Z

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    ax.plot(x, y, z, label='Load Trajectory', color='blue', linewidth=2)
    ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color='green', label='Start', s=100)  # Start point
    ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color='red', label='End', s=100)   # End point

    # Set labels
    ax.set_xlabel('X (Unity Horizontal)')
    ax.set_ylabel('Z (Unity Depth)')
    ax.set_zlabel('Y (Unity Height)')

    # Title and legend
    ax.set_title('3D Load Trajectory Visualization (Unity Coordinates)')
    ax.legend()

    # Show the plot
    plt.show()
else:
    print("Error: The dataset must contain 'X', 'Y', and 'Z' columns.")
