from sklearn.preprocessing import RobustScaler

def _scale_data(Y_data_array, XZ_data_array, **params):
    verbose = params.get("verbose")
    coordinates = params.get("coordinates")
    if verbose:
        print("Scaling the data...")

    for coordinate in coordinates:
        if coordinate == "Y":
            Y_scaler = RobustScaler()
            Y_data_scaled = Y_scaler.fit_transform(Y_data_array)
        if coordinate == "XZ":
            XZ_scaler = RobustScaler()
            XZ_data_scaled = XZ_scaler.fit_transform(XZ_data_array)

    return Y_data_scaled, Y_scaler, XZ_data_scaled, XZ_scaler