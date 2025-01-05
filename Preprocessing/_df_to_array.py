def _df_to_array(df, **params):
    """
    Convert the dataframe to a dictionary of numpy arrays
    for each coordinate group in `coordinates`.

    Example:
      coordinates = ["X", "Z"] => returns a dict {
         "X":  data[:, [0]],
         "Z":  data[:, [2]]
      }
      coordinates = ["XZ"] => returns a dict {
         "XZ": data[:, [0, 2]]
      }
      coordinates = ["XYZ"] => returns a dict {
         "XYZ": data[:, [0, 1, 2]]
      }
      ... etc.
    """
    verbose = params.get("verbose", True)
    coordinates = params.get("coordinates")
    coord_to_indices = params.get("coord_to_indices")

    if verbose:
        print("Converting the dataframe to array(s)...")

    # Extract just X, Y, Z columns as a NumPy array
    # data.shape => (num_samples, 3)
    data = df[['X', 'Y', 'Z']].values

    # We'll store the resulting arrays in a dictionary
    data_arrays_dict = {}

    # For each coordinate string in `coordinates`, slice from data
    for coord_str in coordinates:
        if coord_str not in coord_to_indices:
            raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
                             f"Supported keys: {list(coord_to_indices.keys())}")
        # Retrieve the columns we want
        cols = coord_to_indices[coord_str]
        # Slice out the desired columns
        data_arrays_dict[coord_str] = data[:, cols]

    return data_arrays_dict