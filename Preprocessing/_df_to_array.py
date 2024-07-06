def _df_to_array(df, **params):
    verbose = params.get("verbose")
    coordinates = params.get("coordinates")
    if verbose:
        print("Converting the dataframe to array...")
        
    data = df[['X', 'Y', 'Z']].values

    for coordinate in coordinates:
        if coordinate == 'Y':
            # Extract the Y coordinate
            Y_data_array = data[:, 1:2]
        if coordinate == 'XZ':
            # Extract the combined X and Z coordinates
            XZ_data_array = data[:, [0, 2]]

    return Y_data_array, XZ_data_array 