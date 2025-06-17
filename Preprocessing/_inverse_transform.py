def _inverse_transform(scaler, data, coord_str, **params):
    coord_to_dim = params.get("coord_to_dim")

    if coord_str not in coord_to_dim:
        raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
                         f"Supported keys: {list(coord_to_dim.keys())}")

    num_features = coord_to_dim[coord_str]

    original_shape = data.shape

    data_reshaped = data.reshape(-1, num_features)

    data_inversed_2d = scaler.inverse_transform(data_reshaped)

    data_inversed = data_inversed_2d.reshape(original_shape)

    return data_inversed
