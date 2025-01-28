import numpy as np

def _compute_node_features(sample_dict, t_index, **params):
    """
    sample_dict: e.g. { "X": (sequence_length,1), "Y": (sequence_length,1), "Z": (sequence_length,1),
                        "VX": ..., "VY": ..., ... }
    t_index: which sample index to pick, or we might just pass the slices already

    Returns node_features shape => (sequence_length, D_total)
      if we want X/Y/Z + optional velocity + optional accel.
    """
    use_velocity     = params.get("use_velocity")
    use_acceleration = params.get("use_acceleration")

    # Always get X,Y,Z => shape (sequence_length,1)
    Xpos = sample_dict["X"][t_index]  # for example
    Ypos = sample_dict["Y"][t_index]
    Zpos = sample_dict["Z"][t_index]

    feats_list = [Xpos, Ypos, Zpos]  # each shape => (sequence_length, 1)

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

    # Concatenate along last axis => shape (sequence_length, n_features)
    node_features = np.concatenate(feats_list, axis=1)
    return node_features
