import numpy as np
from sklearn.preprocessing import RobustScaler

# def _scale_data(arrays_list, **params):
#     """
#     arrays_list: list of dicts, each dict => { "X": ..., "Y": ..., "Z": ..., "VX": ..., ... } unscaled
#     We'll fit one scaler per key in params["coordinates"].
#     """
#     use_velocity = params.get("use_velocity")
#     use_acceleration = params.get("use_acceleration")
#     coordinates = params.get("coordinates")
#     if use_velocity:
#         coordinates += ["VX","VY","VZ"]
#     if use_acceleration:
#         coordinates += ["AX","AY","AZ"]
#     verbose = params.get("verbose", True)

#     if verbose:
#         print("Fitting scalers globally for each coordinate/feature & scaling each scenario.")

#     # Gather all data for each coordinate from all scenarios
#     print(arrays_list)
#     raise ValueError
#     big_data = {coord: [] for coord in coordinates}
#     for scenario_dict in arrays_list:
#         for coord in coordinates:
#             big_data[coord].append(scenario_dict[coord])  # each is shape (N,1)

#     # Fit a single scaler for each coordinate
#     scalers_dict = {}
#     for coord in coordinates:
#         cat_data = np.concatenate(big_data[coord], axis=0)  # combine from all scenarios
#         scaler = RobustScaler()
#         scaler.fit(cat_data)
#         scalers_dict[coord] = scaler
#         if verbose:
#             print(f"Fitted scaler for '{coord}' shape={cat_data.shape}.")

#     # Transform each scenario's arrays
#     scaled_arrays_list = []
#     for scenario_dict in arrays_list:
#         scaled_scenario = {}
#         for coord in coordinates:
#             arr = scenario_dict[coord]
#             scaled_scenario[coord] = scalers_dict[coord].transform(arr)
#         scaled_arrays_list.append(scaled_scenario)

#     return scaled_arrays_list, scalers_dict





####################################
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
    coordinates = params.get("coordinates")
    verbose = params.get("verbose", True)

    if verbose:
        print("Fitting scalers globally for each coordinate & scaling each scenario.")

    # Gather all data for each coordinate from all scenarios
    big_data = {coord_str: [] for coord_str in coordinates}
    for scenario_dict in arrays_list:
        for coord_str in coordinates:
            big_data[coord_str].append(scenario_dict[coord_str])
    
    # Fit a single scaler for each coordinate
    scalers_dict = {}
    for coord_str in coordinates:
        cat_data = np.concatenate(big_data[coord_str], axis=0)  # combine rows from all scenarios
        scaler = RobustScaler()
        scaler.fit(cat_data)
        scalers_dict[coord_str] = scaler
        if verbose:
            print(f"Fitted scaler for coordinate '{coord_str}' with data shape {cat_data.shape}.")

    # Transform each scenario's arrays
    scaled_arrays_list = []
    for scenario_dict in arrays_list:
        scaled_scenario = {}
        for coord_str in coordinates:
            arr = scenario_dict[coord_str]
            # scaled_scenario[coord_str] = arr
            scaled_scenario[coord_str] = scalers_dict[coord_str].transform(arr)
        scaled_arrays_list.append(scaled_scenario)

    return scaled_arrays_list, scalers_dict