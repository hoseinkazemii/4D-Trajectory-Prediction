# Inverse transform the predictions and actual values
def _inverse_transform(scaler, data, **params):
    verbose = params.get("verbose")
    if verbose:
        print("inverse transform...")
    data_reshaped = data.reshape(-1, 3)
    data_inversed = scaler.inverse_transform(data_reshaped)
    data = data_inversed.reshape(data.shape)
    return data