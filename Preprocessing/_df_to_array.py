def _df_to_array(df, **params):
    """
    Convert a single scenario's DataFrame into a dict of arrays, one per coordinate string in params["coordinates"].
    Example: 
      If coordinates=["X","Y"], then we return { "X": (N,1), "Y": (N,1) }.
      If coordinates=["XYZ"], we return { "XYZ": (N,3) }.
    """
    coordinates = params.get("coordinates")
    coord_to_indices = params.get("coord_to_indices")
    verbose = params.get("verbose", True)

    if verbose:
        print("Converting single scenario DataFrame to arrays for:", coordinates)

    # Extract raw data => shape (N, 3) from columns ["X","Y","Z"]
    data_3cols = df[["X","Y","Z"]].values

    data_arrays_dict = {}
    for coord_str in coordinates:
        cols = coord_to_indices[coord_str]  # e.g. [0,2] for "XZ"
        data_arrays_dict[coord_str] = data_3cols[:, cols]  # slice the appropriate columns
    return data_arrays_dict

def _dfs_to_array(df_list, **params):
    verbose = params.get("verbose", True)
    if verbose:
        print("Converting each scenario DataFrame to arrays (unscaled).")

    arrays_list = []

    # Convert each DataFrame to arrays (dict form: { "X": <array>, "Y": <array>, ... })
    for df in df_list:
        arrays_for_scenario = _df_to_array(df, **params)
        # This should return a dict, e.g. { "X": shape(N,1), "Y": shape(N,1), "Z": shape(N,1), ... }
        arrays_list.append(arrays_for_scenario)
    
    return arrays_list


def _df_to_array_single(df, coordinates, **params):
    """
    Convert one DataFrame to a dict of arrays, keyed by coordinate string.
    e.g.: { "XYZ": <ndarray>, "XZ": <ndarray>, ... }
    """
    coord_to_indices = params.get("coord_to_indices")

    # Extract the 3 columns from the DF
    data_3cols = df[['X', 'Y', 'Z']].values  # shape (num_rows, 3)

    arrays_dict = {}
    for coord_str in coordinates:
        if coord_str not in coord_to_indices:
            raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
                             f"Supported keys: {list(coord_to_indices.keys())}")

        # Retrieve the columns we want
        col_inds = coord_to_indices[coord_str]
        arrays_dict[coord_str] = data_3cols[:, col_inds]
    return arrays_dict






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
