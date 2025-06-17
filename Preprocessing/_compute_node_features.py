import numpy as np

def _compute_node_features(sample_dict, t_index, **params):
    use_velocity     = params.get("use_velocity")
    use_acceleration = params.get("use_acceleration")

    Xpos = sample_dict["X"][t_index]
    Ypos = sample_dict["Y"][t_index]
    Zpos = sample_dict["Z"][t_index]

    feats_list = [Xpos, Ypos, Zpos]

    if use_velocity:
        VX = sample_dict["VX"][t_index]
        VY = sample_dict["VY"][t_index]
        VZ = sample_dict["VZ"][t_index]
        feats_list.extend([VX, VY, VZ])

    if use_acceleration:
        AX = sample_dict["AX"][t_index]
        AY = sample_dict["AY"][t_index]
        AZ = sample_dict["AZ"][t_index]
        feats_list.extend([AX, AY, AZ])

    node_features = np.concatenate(feats_list, axis=1)
    return node_features
