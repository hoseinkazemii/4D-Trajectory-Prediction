import numpy as np
from sklearn.preprocessing import RobustScaler

def _scale_data(arrays_list, **params):
    use_gnn = params.get("use_gnn")

    if use_gnn:
       scaled_arrays_list, scalers_dict = _scale_data_gnn(arrays_list, **params)
    else:
        scaled_arrays_list, scalers_dict = _scale_data_timeseries(arrays_list, **params)
    
    return scaled_arrays_list, scalers_dict


def _scale_data_gnn(arrays_list, **params):

    use_velocity = params.get("use_velocity")
    use_acceleration = params.get("use_acceleration")
    verbose = params.get("verbose", True)

    if verbose:
        print("[_scale_data] Determining which fields to scale...")

    fields_to_scale = []

    if any("XYZ" in scenario_dict for scenario_dict in arrays_list):
        fields_to_scale.append("XYZ")

    if use_velocity:
        for vfield in ["VX", "VY", "VZ"]:
            if any(vfield in scenario_dict for scenario_dict in arrays_list):
                fields_to_scale.append(vfield)

    if use_acceleration:
        for afield in ["AX", "AY", "AZ"]:
            if any(afield in scenario_dict for scenario_dict in arrays_list):
                fields_to_scale.append(afield)

    if verbose:
        print(f"[_scale_data] Fields we will scale: {fields_to_scale}")

    big_data = {field_str: [] for field_str in fields_to_scale}
    for scenario_dict in arrays_list:
        for field_str in fields_to_scale:
            if field_str in scenario_dict:
                big_data[field_str].append(scenario_dict[field_str])
    
    scalers_dict = {}
    for field_str in fields_to_scale:
        cat_data = np.concatenate(big_data[field_str], axis=0)
        scaler = RobustScaler()
        scaler.fit(cat_data)
        scalers_dict[field_str] = scaler
        if verbose:
            print(f"[ _scale_data ] Fitted scaler for '{field_str}' with concatenated shape: {cat_data.shape}.")

    scaled_arrays_list = []
    for scenario_dict in arrays_list:
        scaled_scenario = {}
        
        for field_str in fields_to_scale:
            if field_str in scenario_dict:
                arr = scenario_dict[field_str]
                scaled_scenario[field_str] = scalers_dict[field_str].transform(arr)
        
        for key, val in scenario_dict.items():
            if key not in fields_to_scale:
                scaled_scenario[key] = val
        
        scaled_arrays_list.append(scaled_scenario)

    return scaled_arrays_list, scalers_dict


def _scale_data_timeseries(arrays_list, **params):
    coordinates = params.get("coordinates")
    verbose = params.get("verbose", True)

    if verbose:
        print("Fitting scalers globally for each coordinate & scaling each scenario.")

    big_data = {coord_str: [] for coord_str in coordinates}
    for scenario_dict in arrays_list:
        for coord_str in coordinates:
            big_data[coord_str].append(scenario_dict[coord_str])
    
    scalers_dict = {}
    for coord_str in coordinates:
        cat_data = np.concatenate(big_data[coord_str], axis=0)
        scaler = RobustScaler()
        scaler.fit(cat_data)
        scalers_dict[coord_str] = scaler
        if verbose:
            print(f"Fitted scaler for coordinate '{coord_str}' with data shape {cat_data.shape}.")

    scaled_arrays_list = []
    for scenario_dict in arrays_list:
        scaled_scenario = {}
        for coord_str in coordinates:
            arr = scenario_dict[coord_str]
            scaled_scenario[coord_str] = scalers_dict[coord_str].transform(arr)
        scaled_arrays_list.append(scaled_scenario)

    return scaled_arrays_list, scalers_dict
