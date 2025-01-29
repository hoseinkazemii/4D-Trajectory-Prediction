import numpy as np
from sklearn.preprocessing import RobustScaler

def _scale_data(arrays_list, use_velocity=False, use_acceleration=False, use_gnn=False, **params):
    """
    arrays_list: a list of dicts, each dict => { possibly "XYZ", "VX", "VY", "VZ", "AX", "AY", "AZ", etc. }
    
    We'll gather which fields should be scaled based on the flags:
      - use_velocity or use_gnn => scale velocity fields if present
      - use_acceleration or use_gnn => scale acceleration fields if present
      
    Then, for each chosen field, gather all data across ALL scenarios, 
    fit a single scaler, and transform each scenario's arrays.
    
    Returns: (scaled_arrays_list, scalers_dict)
      - scaled_arrays_list => same shape as arrays_list but with scaled values
      - scalers_dict => { field_str: fitted_scaler }
    """
    verbose = params.get("verbose", True)

    if verbose:
        print("[_scale_data] Determining which fields to scale...")

    # Base list: always scale "XYZ" if present
    fields_to_scale = []
    # We'll always try "XYZ" (assuming positional data is always relevant)
    # Feel free to omit "XYZ" if you only want to conditionally scale it. 
    # But in your previous example, "XYZ" is definitely scaled.
    if any("XYZ" in scenario_dict for scenario_dict in arrays_list):
        fields_to_scale.append("XYZ")

    # If velocity or GNN is used, we scale velocity fields if they exist
    if use_velocity or use_gnn:
        for vfield in ["VX", "VY", "VZ"]:
            if any(vfield in scenario_dict for scenario_dict in arrays_list):
                fields_to_scale.append(vfield)
                
    # If acceleration or GNN is used, we scale acceleration fields if they exist
    if use_acceleration or use_gnn:
        for afield in ["AX", "AY", "AZ"]:
            if any(afield in scenario_dict for scenario_dict in arrays_list):
                fields_to_scale.append(afield)
    
    if verbose:
        print(f"[_scale_data] Fields we will scale: {fields_to_scale}")

    # Gather all data for each field from all scenarios
    big_data = {field_str: [] for field_str in fields_to_scale}
    for scenario_dict in arrays_list:
        for field_str in fields_to_scale:
            if field_str in scenario_dict:  # check presence
                big_data[field_str].append(scenario_dict[field_str])
    
    # Fit a single scaler for each field
    scalers_dict = {}
    for field_str in fields_to_scale:
        cat_data = np.concatenate(big_data[field_str], axis=0)  # combine rows from all scenarios
        scaler = RobustScaler()
        scaler.fit(cat_data)
        scalers_dict[field_str] = scaler
        if verbose:
            print(f"[ _scale_data ] Fitted scaler for '{field_str}' with concatenated shape: {cat_data.shape}.")

    # Transform each scenario's arrays
    scaled_arrays_list = []
    for scenario_dict in arrays_list:
        scaled_scenario = {}
        
        # For any field we scale, do the transform if it exists
        for field_str in fields_to_scale:
            if field_str in scenario_dict:
                arr = scenario_dict[field_str]
                scaled_scenario[field_str] = scalers_dict[field_str].transform(arr)
        
        # For fields we are NOT scaling (like we only scale them if they exist 
        # but the flags do not ask for them), copy them over as is:
        # Or you can skip them entirely, depending on your needs.
        for key, val in scenario_dict.items():
            if key not in fields_to_scale:
                scaled_scenario[key] = val
        
        scaled_arrays_list.append(scaled_scenario)

    return scaled_arrays_list, scalers_dict






####################################
# import numpy as np
# from sklearn.preprocessing import RobustScaler

# def _scale_data(arrays_list, **params):
#     """
#     arrays_list: a list of dicts, each dict => { coord_str: unscaled_array }
#     e.g. arrays_list[i]["X"] => shape (rows_i,1)

#     We'll fit one scaler per coord_str across ALL scenarios, 
#     then transform each scenario's arrays.

#     Returns: (scaled_arrays_list, scalers_dict)
#       - scaled_arrays_list => same shape as arrays_list but with scaled values
#       - scalers_dict => { coord_str: fitted_scaler }
#     """
#     coordinates = params.get("coordinates")
#     verbose = params.get("verbose", True)

#     if verbose:
#         print("Fitting scalers globally for each coordinate & scaling each scenario.")

#     # Gather all data for each coordinate from all scenarios
#     big_data = {coord_str: [] for coord_str in coordinates}
#     for scenario_dict in arrays_list:
#         for coord_str in coordinates:
#             big_data[coord_str].append(scenario_dict[coord_str])
    
#     # Fit a single scaler for each coordinate
#     scalers_dict = {}
#     for coord_str in coordinates:
#         cat_data = np.concatenate(big_data[coord_str], axis=0)  # combine rows from all scenarios
#         scaler = RobustScaler()
#         scaler.fit(cat_data)
#         scalers_dict[coord_str] = scaler
#         if verbose:
#             print(f"Fitted scaler for coordinate '{coord_str}' with data shape {cat_data.shape}.")

#     # Transform each scenario's arrays
#     scaled_arrays_list = []
#     for scenario_dict in arrays_list:
#         scaled_scenario = {}
#         for coord_str in coordinates:
#             arr = scenario_dict[coord_str]
#             # scaled_scenario[coord_str] = arr
#             scaled_scenario[coord_str] = scalers_dict[coord_str].transform(arr)
#         scaled_arrays_list.append(scaled_scenario)

#     return scaled_arrays_list, scalers_dict