from sklearn.preprocessing import MinMaxScaler


def _scale_data(data, **params):
    verbose = params.get("verbose")
    if verbose:
        print("scaling the data...")

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    return data, scaler