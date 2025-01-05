def _inverse_transform(scaler, data, coord_str, **params):
    """
    Inverse-transforms the scaled data (predictions or ground truths) back to the original range.

    scaler: a fitted scaler (e.g., RobustScaler, MinMaxScaler, etc.)
    data: numpy array of shape (num_samples, prediction_horizon, num_features) 
          or (num_samples, sequence_length, num_features),
          or some 2D shape if this is the output of the model, etc.
    coord_str: string indicating which coordinate or coordinate group (e.g. "X", "Y", "Z", "XZ", "XYZ")

    Returns:
      data_inversed: the same shape as 'data', but with values transformed to the original scale.
    """

    # A small dictionary mapping coordinate strings to their dimension
    coord_to_dim = params.get("coord_to_dim")

    if coord_str not in coord_to_dim:
        raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
                         f"Supported keys: {list(coord_to_dim.keys())}")

    num_features = coord_to_dim[coord_str]

    # Save original shape so we can reshape back
    original_shape = data.shape

    # Reshape to 2D => (num_samples * time, num_features)
    # e.g., if data = (batch_size, horizon, num_features), we flatten the first two dims
    data_reshaped = data.reshape(-1, num_features)

    # Apply inverse transform
    data_inversed_2d = scaler.inverse_transform(data_reshaped)

    # Reshape back to original
    data_inversed = data_inversed_2d.reshape(original_shape)

    return data_inversed
