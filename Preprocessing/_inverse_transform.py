def _inverse_transform(scaler, data, coordinate, **params):
    # Determine the number of features based on the coordinate
    if coordinate == 'Y':
        num_features = 1
        # Reshape data to the shape expected by the scaler
        original_shape = data.shape
        data_reshaped = data.reshape(-1, num_features)
        data_inversed = scaler.inverse_transform(data_reshaped)
        # Reshape back to the original shape
        data_inversed = data_inversed.reshape(original_shape)

    elif coordinate == 'XZ':
        num_features = 2
        original_shape = data.shape
        data_reshaped = data.reshape(-1, num_features)
        data_inversed = scaler.inverse_transform(data_reshaped)
        data_inversed = data_inversed.reshape(original_shape)

    return data_inversed