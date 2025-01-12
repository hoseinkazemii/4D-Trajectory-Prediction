# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Load data
# df = pd.read_csv('LoadData_20240624151409.csv')

# # Sample coordinates list; change as needed
# coordinates = ["X","Y","Z"]

# def euclidean_distance(points):
#     """
#     Computes Euclidean distances between consecutive points in a 2D NumPy array.
#     Each row in `points` is a point in space.
#     Returns an array of distances between consecutive points.
#     """
#     # Calculate differences between consecutive points
#     diffs = np.diff(points, axis=0)
#     # Euclidean distance for each consecutive pair
#     distances = np.linalg.norm(diffs, axis=1)
#     return distances

# def visualize_velocities(df, coordinates):
#     time = df['Time'].values
    
#     # Use np.diff on time to handle potentially non-uniform time intervals
#     time_diffs = np.diff(time)
#     # For constant time intervals, this will be an array of ones.
    
#     for group in coordinates:
#         cols = list(group)  # For "YZ" -> ['Y', 'Z'], etc.
#         if not all(col in df.columns for col in cols):
#             print(f"Columns for group {group} not found in data.")
#             continue

#         # Extract the coordinate values as a NumPy array
#         points = df[cols].values

#         # Compute Euclidean distances between consecutive points
#         distances = euclidean_distance(points)

#         # Compute velocities by dividing distance by time difference at each interval.
#         # Note: The length of `distances` is one less than the length of `time_diffs`.
#         # We'll align them by using time_diffs corresponding to these intervals.
#         velocities = distances / time_diffs

#         # Time points corresponding to these velocity measurements
#         # They represent the mid-point between consecutive time intervals
#         velocity_time = (time[:-1] + time[1:]) / 2

#         # Plot the velocity over time for the coordinate group
#         plt.figure(figsize=(10, 4))
#         plt.plot(velocity_time, velocities, label=f'Velocity for {group}')
#         plt.title(f'Velocity Over Time for {group}')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Velocity')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(f"test1_{coordinates}.png",format="png",dpi=300)
#         plt.show()

# visualize_velocities(df, coordinates)



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Load data
# df = pd.read_csv('LoadData_20240624151409.csv')

# # Sample coordinates list; change as needed
# coordinates = ["X","Y","Z"]

# def euclidean_distance(points):
#     """
#     Computes Euclidean distances between consecutive points in a 2D NumPy array.
#     Each row in `points` is a point in space.
#     Returns an array of distances between consecutive points.
#     """
#     # Calculate differences between consecutive points
#     diffs = np.diff(points, axis=0)
#     # Euclidean distance for each consecutive pair
#     distances = np.linalg.norm(diffs, axis=1)
#     return distances

# def visualize_velocities(df, coordinates):
#     time = df['Time'].values
    
#     # Use np.diff on time to handle potentially non-uniform time intervals
#     time_diffs = np.diff(time)
#     # For constant time intervals, this will be an array of ones.
    
#     for group in coordinates:
#         cols = list(group)  # For "YZ" -> ['Y', 'Z'], etc.
#         if not all(col in df.columns for col in cols):
#             print(f"Columns for group {group} not found in data.")
#             continue

#         # Extract the coordinate values as a NumPy array
#         points = df[cols].values

#         # Compute Euclidean distances between consecutive points
#         distances = euclidean_distance(points)

#         # Compute velocities by dividing distance by time difference at each interval.
#         # Note: The length of `distances` is one less than the length of `time_diffs`.
#         # We'll align them by using time_diffs corresponding to these intervals.
#         velocities = distances / time_diffs

#         # Time points corresponding to these velocity measurements
#         # They represent the mid-point between consecutive time intervals
#         velocity_time = (time[:-1] + time[1:]) / 2

#         # Plot the velocity over time for the coordinate group
#         plt.figure(figsize=(10, 4))
#         plt.plot(velocity_time, velocities, label=f'Velocity for {group}')
#         plt.title(f'Velocity Over Time for {group}')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Velocity')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(f"Velocity_Time_Plot_{group}_Coordinate(s).png", format="png", dpi=300)
#         plt.show()

# # Call the visualization function
# visualize_velocities(df, coordinates)


#########################
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Load data
# df = pd.read_csv('LoadData_20240624151409.csv')

# # Sample coordinates list; change as needed
# coordinates = ["XYZ"]

# def euclidean_distance(points):
#     """
#     Computes Euclidean distances between consecutive points in a 2D NumPy array.
#     Each row in `points` is a point in space.
#     Returns an array of distances between consecutive points.
#     """
#     diffs = np.diff(points, axis=0)
#     distances = np.linalg.norm(diffs, axis=1)
#     return distances

# def theoretical_velocity(velocity_time, group):
#     """
#     Placeholder function to compute theoretical velocity for a given coordinate group over time.
#     Customize this function with the actual kinematic equations you expect.
#     """
#     # Example placeholder formulas:
#     if group == "X":
#         # Assume constant acceleration along X for demonstration:
#         v0 = 0.0        # initial velocity (m/s)
#         a = 1.0         # constant acceleration (m/s^2)
#         return v0 + a * (velocity_time - velocity_time[0])
    
#     elif group == "YZ":
#         # For demonstration, assume constant velocity for YZ direction:
#         const_velocity = 2.0
#         return np.full_like(velocity_time, const_velocity)
    
#     # Add more conditions for other coordinate groups as needed.
#     else:
#         # Default: constant velocity of 0
#         return np.zeros_like(velocity_time)

# def visualize_velocities_with_theory(df, coordinates):
#     time = df['Time'].values
#     # Compute time differences to handle non-uniform intervals
#     time_diffs = np.diff(time)
    
#     for group in coordinates:
#         cols = list(group)  # e.g., "YZ" -> ['Y', 'Z']
#         if not all(col in df.columns for col in cols):
#             print(f"Columns for group {group} not found in data.")
#             continue

#         # Extract points for the coordinate group
#         points = df[cols].values

#         # Compute Euclidean distances and velocities
#         distances = euclidean_distance(points)
#         velocities = distances / time_diffs

#         # Time points for velocity measurements (midpoints)
#         velocity_time = (time[:-1] + time[1:]) / 2

#         # Compute theoretical velocity for the group at the given times
#         theor_vel = theoretical_velocity(velocity_time, group)

#         # Plot actual vs. theoretical velocity
#         plt.figure(figsize=(10, 5))
#         plt.plot(velocity_time, velocities, label=f'Actual Velocity for {group}')
#         plt.plot(velocity_time, theor_vel, label=f'Theoretical Velocity for {group}', linestyle='--')
#         plt.title(f'Actual vs. Theoretical Velocity Over Time for {group}')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Velocity')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

# # Call the visualization function
# visualize_velocities_with_theory(df, coordinates)





###################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

########################################################
# 1) Load CSV
########################################################
# Replace 'trajectory_data.csv' with your actual CSV filename
df = pd.read_csv('LoadData_20240624151409.csv')  

# Make sure your CSV has columns: Time, X, Y, Z
time = df['Time'].values  # e.g., [65.44, 66.44, 67.44, ...]

########################################################
# 2) Define helper functions
########################################################
def get_coordinate_data(df, coord):
    """
    Returns a 1D array corresponding to the chosen coordinate combination.
    For multi-dimensional combos (e.g., 'XY'), we return the magnitude 
    in that plane, e.g., sqrt(X^2 + Y^2) for 'XY'.
    You can customize this function if you want a different approach.
    """
    if coord == 'X':
        return df['X'].values
    elif coord == 'Y':
        return df['Y'].values
    elif coord == 'Z':
        return df['Z'].values
    
    # 2D combinations (magnitudes):
    elif coord == 'XY':
        return np.sqrt(df['X']**2 + df['Y']**2)
    elif coord == 'XZ':
        return np.sqrt(df['X']**2 + df['Z']**2)
    elif coord == 'YZ':
        return np.sqrt(df['Y']**2 + df['Z']**2)
    
    # 3D combination (magnitude):
    elif coord == 'XYZ':
        return np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
    
    else:
        raise ValueError(f"Coordinate combination '{coord}' not recognized.")

def compute_velocity_acceleration(time_array, position_array):
    """
    Given time_array (e.g., [t0, t1, t2, ...]) and position_array 
    (e.g., [x0, x1, x2, ...]), compute velocity & acceleration 
    assuming uniform time steps. 
    
    If the time steps are *exactly* 1 second, then:
        velocity[i]   ~ position[i+1] - position[i]
        acceleration[i] ~ velocity[i+1] - velocity[i]
    More generally:
        velocity[i] = (pos[i+1] - pos[i]) / (time[i+1] - time[i])
        acceleration[i] = (vel[i+1] - vel[i]) / (time[i+1] - time[i])
    """
    dt = np.diff(time_array)  # [t1-t0, t2-t1, ...]
    
    # Velocity: derivative of position w.r.t. time
    velocity = np.diff(position_array) / dt  # shape: (N-1,)
    
    # Acceleration: derivative of velocity w.r.t. time
    # We'll use dt[1:] because acceleration is one step shorter than velocity
    acceleration = np.diff(velocity) / dt[1:]  # shape: (N-2,)

    return velocity, acceleration

def custom_loss(vel_true, acc_true, vel_pred, acc_pred):
    """
    Example of a custom loss function that penalizes predictions 
    that do not match the observed velocities and accelerations.
    
    For demonstration, we use a simple mean squared error:
        loss_v = mean((vel_true - vel_pred)^2)
        loss_a = mean((acc_true - acc_pred)^2)
    total_loss = loss_v + loss_a
    
    You can customize this to further incorporate any relationship 
    you want between v/t and a. For instance:
        loss_relationship = mean((vel_pred - acc_pred)**2)
    and then combine it with the errors.
    """
    loss_v = np.mean((vel_true - vel_pred)**2)
    loss_a = np.mean((acc_true - acc_pred)**2)
    
    # Optionally, add a penalty for differences in the v/a relationship:
    # e.g., if time step = 1, then v ~ a
    loss_relationship = np.mean((vel_pred - acc_pred)**2)
    
    total_loss = loss_v + loss_a + loss_relationship
    return total_loss

########################################################
# 3) Visualization for various coordinate combos
########################################################
# Example coordinate lists; you can replace or loop over these
coordinate_sets = [
    ["X", "YZ"],    # custom coordinates
    ["X", "Y", "Z"],
    ["XY", "Z"],
    ["XYZ"]
]

for coordinates in coordinate_sets:
    print("Visualizing coordinate set:", coordinates)
    
    # For each combination in the current set, compute velocity & acceleration:
    for combo in coordinates:
        # Extract the 1D magnitude data (X, YZ, XY, etc.)
        data = get_coordinate_data(df, combo)

        # Compute velocity & acceleration
        vel, acc = compute_velocity_acceleration(time, data)
        
        # Create time arrays that align with vel & acc
        # vel is defined between time[0] and time[-1], so let's define:
        time_vel = 0.5 * (time[:-1] + time[1:])   # midpoints for velocity
        time_acc = 0.5 * (time_vel[:-1] + time_vel[1:])  # midpoints for acceleration

        # Plot velocity and acceleration
        plt.figure(figsize=(8, 4))
        plt.plot(time_vel, vel, label='Velocity')
        plt.plot(time_acc, acc, label='Acceleration')
        plt.title(f"Velocity & Acceleration ({combo})")
        plt.xlabel("Time (s)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

########################################################
# 4) (Optional) Example usage of the custom loss
########################################################
# Suppose you have a model that predicts velocity & acceleration.
# Here we just create dummy predictions to show usage.

for coordinates in ["X", "XYZ"]:
    data = get_coordinate_data(df, coordinates)
    vel_true, acc_true = compute_velocity_acceleration(time, data)

    # Let's say your model predicted something; here we just offset it for demo
    vel_pred = vel_true + np.random.normal(0, 0.1, size=vel_true.shape)
    acc_pred = acc_true + np.random.normal(0, 0.1, size=acc_true.shape)

    # Compute the loss
    loss_value = custom_loss(vel_true[:len(acc_true)], 
                             acc_true, 
                             vel_pred[:len(acc_true)], 
                             acc_pred)

    print(f"Custom loss for coordinate '{coordinates}' = {loss_value:.4f}")
