import numpy as np
from sklearn.preprocessing import RobustScaler

def _scale_data(arrays_list, **params):
    """
    arrays_list: a list of dicts, each dict => { coord_str: unscaled_array }
    e.g. arrays_list[i]["X"] => shape (rows_i,1)

    We'll fit one scaler per coord_str across ALL scenarios, 
    then transform each scenario's arrays.

    Returns: (scaled_arrays_list, scalers_dict)
      - scaled_arrays_list => same shape as arrays_list but with scaled values
      - scalers_dict => { coord_str: fitted_scaler }
    """
    coordinates = params.get("coordinates", [])
    verbose = params.get("verbose", True)

    if verbose:
        print("Fitting scalers globally for each coordinate & scaling each scenario.")

    # 1) Gather all data for each coordinate from all scenarios
    big_data = {coord_str: [] for coord_str in coordinates}
    for scenario_dict in arrays_list:
        for coord_str in coordinates:
            big_data[coord_str].append(scenario_dict[coord_str])
    
    # 2) Fit a single scaler for each coordinate
    scalers_dict = {}
    for coord_str in coordinates:
        cat_data = np.concatenate(big_data[coord_str], axis=0)  # combine rows from all scenarios
        scaler = RobustScaler()
        scaler.fit(cat_data)
        scalers_dict[coord_str] = scaler
        if verbose:
            print(f"Fitted scaler for coordinate '{coord_str}' with data shape {cat_data.shape}.")

    # 3) Transform each scenario's arrays
    scaled_arrays_list = []
    for scenario_dict in arrays_list:
        scaled_scenario = {}
        for coord_str in coordinates:
            arr = scenario_dict[coord_str]
            scaled_scenario[coord_str] = scalers_dict[coord_str].transform(arr)
        scaled_arrays_list.append(scaled_scenario)

    return scaled_arrays_list, scalers_dict


# from sklearn.preprocessing import RobustScaler

# def _scale_data(data_arrays_dict, **params):
#     verbose = params.get("verbose", True)
#     if verbose:
#         print("Scaling the data with RobustScaler...")

#     scaled_arrays_dict = {}
#     scalers_dict = {}

#     # Loop over each coordinate key and array
#     for coord_str, arr in data_arrays_dict.items():
#         # Create a RobustScaler for this particular coordinate or coordinate-group
#         scaler = RobustScaler()
#         # Fit_transform the array
#         arr_scaled = scaler.fit_transform(arr)

#         # Store scaled array
#         scaled_arrays_dict[coord_str] = arr_scaled
#         # Store the scaler for possible inverse transforms later
#         scalers_dict[coord_str] = scaler

#     return scaled_arrays_dict, scalers_dict