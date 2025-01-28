import numpy as np

def _df_to_array(df, **params):
    """Convert DataFrame to arrays with position, velocity, and acceleration."""
    coordinates = params.get("coordinates")
    coord_to_indices = params.get("coord_to_indices")
    use_velocity = params.get("use_velocity")
    use_acceleration = params.get("use_acceleration")
    verbose = params.get("verbose", True)

    if verbose:
        print("Converting DataFrame to arrays for: ", coordinates)

    # Extract position data
    data_3cols = df[["X","Y","Z"]].values
    data_arrays_dict = {}

    # Process each coordinate set (e.g., "XYZ", "XY", etc.)
    for coord_str in coordinates:
        cols = coord_to_indices[coord_str]
        data_arrays_dict[coord_str] = data_3cols[:, cols]

    # Calculate velocity and acceleration if requested
    if use_velocity or use_acceleration:
        time_np = df["Time"].values
        vx_np = np.zeros_like(data_3cols[:, 0])
        vy_np = np.zeros_like(data_3cols[:, 1])
        vz_np = np.zeros_like(data_3cols[:, 2])

        # Calculate velocities
        for t in range(len(time_np) - 1):
            dt = time_np[t+1] - time_np[t]
            dt = max(dt, 1e-6)  # Avoid division by zero
            vx_np[t+1] = (data_3cols[t+1, 0] - data_3cols[t, 0]) / dt
            vy_np[t+1] = (data_3cols[t+1, 1] - data_3cols[t, 1]) / dt
            vz_np[t+1] = (data_3cols[t+1, 2] - data_3cols[t, 2]) / dt

        if use_velocity:
            data_arrays_dict["VX"] = vx_np.reshape(-1, 1)
            data_arrays_dict["VY"] = vy_np.reshape(-1, 1)
            data_arrays_dict["VZ"] = vz_np.reshape(-1, 1)

        if use_acceleration:
            ax_np = np.zeros_like(vx_np)
            ay_np = np.zeros_like(vy_np)
            az_np = np.zeros_like(vz_np)

            # Calculate accelerations
            for t in range(1, len(time_np) - 1):
                dt = time_np[t+1] - time_np[t]
                dt = max(dt, 1e-6)
                ax_np[t+1] = (vx_np[t+1] - vx_np[t]) / dt
                ay_np[t+1] = (vy_np[t+1] - vy_np[t]) / dt
                az_np[t+1] = (vz_np[t+1] - vz_np[t]) / dt

            data_arrays_dict["AX"] = ax_np.reshape(-1, 1)
            data_arrays_dict["AY"] = ay_np.reshape(-1, 1)
            data_arrays_dict["AZ"] = az_np.reshape(-1, 1)

    return data_arrays_dict

def _dfs_to_array(df_list, **params):
    """Process multiple DataFrames to arrays including position, velocity, and acceleration."""
    verbose = params.get("verbose", True)
    if verbose:
        print("Converting DataFrames to arrays with kinematics.")

    arrays_list = []

    for df in df_list:
        arrays_for_scenario = _df_to_array(df, **params)
        arrays_list.append(arrays_for_scenario)

    return arrays_list





################################
# def _df_to_array(df, **params):
#     """
#     Convert a single scenario's DataFrame into a dict of arrays, one per coordinate string in params["coordinates"].
#     Example: 
#       If coordinates=["X","Y"], then we return { "X": (N,1), "Y": (N,1) }.
#       If coordinates=["XYZ"], we return { "XYZ": (N,3) }.
#     """
#     coordinates = params.get("coordinates")
#     coord_to_indices = params.get("coord_to_indices")
#     verbose = params.get("verbose", True)

#     if verbose:
#         print("Converting single scenario DataFrame to arrays for: ", coordinates)

#     # Extract raw data => shape (N, 3) from columns ["X","Y","Z"]
#     data_3cols = df[["X","Y","Z"]].values

#     data_arrays_dict = {}
#     for coord_str in coordinates:
#         cols = coord_to_indices[coord_str]  # e.g. [0,2] for "XZ"
#         data_arrays_dict[coord_str] = data_3cols[:, cols]  # slice the appropriate columns
#     return data_arrays_dict

# def _dfs_to_array(df_list, **params):
#     verbose = params.get("verbose", True)
#     if verbose:
#         print("Converting each scenario DataFrame to arrays (unscaled).")

#     arrays_list = []

#     # Convert each DataFrame to arrays (dict form: { "X": <array>, "Y": <array>, ... })
#     for df in df_list:
#         arrays_for_scenario = _df_to_array(df, **params)
#         # This should return a dict, e.g. { "X": shape(N,1), "Y": shape(N,1), "Z": shape(N,1), ... }
#         arrays_list.append(arrays_for_scenario)
    
#     return arrays_list
###################################

# def _df_to_array_single(df, coordinates, **params):
#     """
#     Convert one DataFrame to a dict of arrays, keyed by coordinate string.
#     e.g.: { "XYZ": <ndarray>, "XZ": <ndarray>, ... }
#     """
#     coord_to_indices = params.get("coord_to_indices")

#     # Extract the 3 columns from the DF
#     data_3cols = df[['X', 'Y', 'Z']].values  # shape (num_rows, 3)

#     arrays_dict = {}
#     for coord_str in coordinates:
#         if coord_str not in coord_to_indices:
#             raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
#                              f"Supported keys: {list(coord_to_indices.keys())}")

#         # Retrieve the columns we want
#         col_inds = coord_to_indices[coord_str]
#         arrays_dict[coord_str] = data_3cols[:, col_inds]
#     return arrays_dict


# def _df_to_array(df, **params):
#     """
#     Convert the dataframe to a dictionary of numpy arrays
#     for each coordinate group in `coordinates`.

#     Example:
#       coordinates = ["X", "Z"] => returns a dict {
#          "X":  data[:, [0]],
#          "Z":  data[:, [2]]
#       }
#       coordinates = ["XZ"] => returns a dict {
#          "XZ": data[:, [0, 2]]
#       }
#       coordinates = ["XYZ"] => returns a dict {
#          "XYZ": data[:, [0, 1, 2]]
#       }
#       ... etc.
#     """
#     verbose = params.get("verbose", True)
#     coordinates = params.get("coordinates")
#     coord_to_indices = params.get("coord_to_indices")

#     if verbose:
#         print("Converting the dataframe to array(s)...")

#     # Extract just X, Y, Z columns as a NumPy array
#     # data.shape => (num_samples, 3)
#     data = df[['X', 'Y', 'Z']].values

#     # We'll store the resulting arrays in a dictionary
#     data_arrays_dict = {}

#     # For each coordinate string in `coordinates`, slice from data
#     for coord_str in coordinates:
#         if coord_str not in coord_to_indices:
#             raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
#                              f"Supported keys: {list(coord_to_indices.keys())}")
#         # Retrieve the columns we want
#         cols = coord_to_indices[coord_str]
#         # Slice out the desired columns
#         data_arrays_dict[coord_str] = data[:, cols]

#     return data_arrays_dict
