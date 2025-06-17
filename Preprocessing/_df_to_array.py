import numpy as np

def _df_to_array(df, **params):
    coordinates     = params.get("coordinates")
    coord_to_indices = params.get("coord_to_indices")
    verbose         = params.get("verbose")
    noise_std       = params.get("noise_std")
    use_velocity               = params.get("use_velocity")
    use_acceleration           = params.get("use_acceleration")
    use_init_final_positions   = params.get("use_init_final_positions")

    if verbose:
        print(f"\nConverting DataFrame to arrays. "
              f"Velocity={use_velocity}, Accel={use_acceleration}, "
              f"Init/Final={use_init_final_positions}, Noise={noise_std}")

    data_3cols = df[["X", "Y", "Z"]].values

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=data_3cols.shape)
        data_3cols += noise

    feats_list = [data_3cols]

    time_np = df["Time"].values

    if use_velocity or use_acceleration:
        vx_np = np.zeros_like(time_np, dtype=np.float32)
        vy_np = np.zeros_like(time_np, dtype=np.float32)
        vz_np = np.zeros_like(time_np, dtype=np.float32)

        for t in range(len(time_np) - 1):
            dt = time_np[t+1] - time_np[t]
            dt = max(dt, 1e-6)
            vx_np[t+1] = (data_3cols[t+1, 0] - data_3cols[t, 0]) / dt
            vy_np[t+1] = (data_3cols[t+1, 1] - data_3cols[t, 1]) / dt
            vz_np[t+1] = (data_3cols[t+1, 2] - data_3cols[t, 2]) / dt

        if use_velocity:
            velocity_array = np.column_stack([vx_np, vy_np, vz_np])
            feats_list.append(velocity_array)

        if use_acceleration:
            ax_np = np.zeros_like(time_np, dtype=np.float32)
            ay_np = np.zeros_like(time_np, dtype=np.float32)
            az_np = np.zeros_like(time_np, dtype=np.float32)

            for t in range(len(time_np) - 1):
                dt = time_np[t+1] - time_np[t]
                dt = max(dt, 1e-6)
                ax_np[t+1] = (vx_np[t+1] - vx_np[t]) / dt
                ay_np[t+1] = (vy_np[t+1] - vy_np[t]) / dt
                az_np[t+1] = (vz_np[t+1] - vz_np[t]) / dt

            accel_array = np.column_stack([ax_np, ay_np, az_np])
            feats_list.append(accel_array)

    if use_init_final_positions:
        x_init, y_init, z_init = data_3cols[0, :]
        x_final, y_final, z_final = data_3cols[-1, :]

        repeated_init  = np.tile([x_init,  y_init,  z_init],  (len(data_3cols), 1))
        repeated_final = np.tile([x_final, y_final, z_final], (len(data_3cols), 1))

        init_final_array = np.hstack([repeated_init, repeated_final])
        feats_list.append(init_final_array)

    final_data = np.hstack(feats_list)

    if verbose:
        print(f"Final feature array shape: {final_data.shape}")

    data_arrays_dict = {}
    for coord_str in coordinates:
        if coord_str not in coord_to_indices:
            raise KeyError(f"'{coord_str}' not found in coord_to_indices. "
                           f"Please define its column indices in main code.")
        cols = coord_to_indices[coord_str]
        data_arrays_dict[coord_str] = final_data[:, cols]

    return data_arrays_dict


def _dfs_to_array(df_list, **params):
    verbose = params.get("verbose", True)
    if verbose:
        print("Converting DataFrames to arrays with kinematics.")

    arrays_list = []

    for df in df_list:
        arrays_for_scenario = _df_to_array(df, **params)
        arrays_list.append(arrays_for_scenario)

    return arrays_list
