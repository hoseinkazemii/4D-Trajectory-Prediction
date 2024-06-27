# Inverse transform the predictions and actual values
def _inverse_transform(scaler, data, coordinate, **params):
    verbose = params.get("verbose")
    if verbose:
        print("inverse transform...")

    # Determine the number of features based on the coordinate
    if coordinate == 'Y':
        num_features = 1
    else:
        num_features = 2

    # Reshape data to the shape expected by the scaler
    data_reshaped = data.reshape(-1, num_features)
    data_inversed = scaler.inverse_transform(data_reshaped)
    
    # Reshape back to the original shape
    data = data_inversed.reshape(data.shape)
    
    return data