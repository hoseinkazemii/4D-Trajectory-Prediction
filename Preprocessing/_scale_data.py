from sklearn.preprocessing import RobustScaler

def _scale_data(data_arrays_dict, **params):
    verbose = params.get("verbose", True)
    if verbose:
        print("Scaling the data with RobustScaler...")

    scaled_arrays_dict = {}
    scalers_dict = {}

    # Loop over each coordinate key and array
    for coord_str, arr in data_arrays_dict.items():
        # Create a RobustScaler for this particular coordinate or coordinate-group
        scaler = RobustScaler()
        # Fit_transform the array
        arr_scaled = scaler.fit_transform(arr)

        # Store scaled array
        scaled_arrays_dict[coord_str] = arr_scaled
        # Store the scaler for possible inverse transforms later
        scalers_dict[coord_str] = scaler

    return scaled_arrays_dict, scalers_dict