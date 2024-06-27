from sklearn.preprocessing import RobustScaler

def _scale_data(data, **params):
    verbose = params.get("verbose")
    if verbose:
        print("scaling the data using RobustScaler...")

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler