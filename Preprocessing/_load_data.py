import os
import numpy as np
import pandas as pd


_LUT = {
    (0, 1):  1, (1, 3):  1, (3, 2):  1, (2, 0):  1,
    (1, 0): -1, (3, 1): -1, (2, 3): -1, (0, 2): -1
}

def _decode_quadrature(df: pd.DataFrame, col_A: str, col_B: str) -> np.ndarray:
    state      = (df[col_A].astype(int) << 1) | df[col_B].astype(int)
    prev_state = state.shift(1).fillna(state.iloc[0]).astype(int)

    delta = np.fromiter(
        (_LUT.get((p, c), 0) for p, c in zip(prev_state, state)),
        dtype=int, count=len(state)
    )

    return np.cumsum(delta)

def _cartesian_from_encoders(df: pd.DataFrame) -> pd.DataFrame:
    theta_cnt  = _decode_quadrature(df,
                    'SignalASlewingAngleRotaryEncoder',
                    'SignalBSlewingAngleRotaryEncoder')

    radius_cnt = _decode_quadrature(df,
                    'SignalARadiusLinearEncoder',
                    'SignalBRadiusLinearEncoder')

    height_cnt = _decode_quadrature(df,
                    'SignalAHookHeightLinearEncoder',
                    'SignalBHookHeightLinearEncoder')

    theta_rad = (2 * np.pi * theta_cnt) / 1024.0
    radius_m  = radius_cnt * 0.005
    height_m  = height_cnt * 0.005

    X = radius_m * np.cos(theta_rad)
    Y = radius_m * np.sin(theta_rad)
    Z = height_m

    return pd.DataFrame(
        {'Time': df['Time'].values,
         'X':    X,
         'Y':    Y,
         'Z':    Z}
    )

def _signal_based_extraction(df, **params):
    time_interval = params.get("time_interval")

    baseline_time = df.iloc[0]['Time']
    new_rows = []
    last_time = -float('inf')

    while baseline_time <= df['Time'].iloc[-1]:
        subset = df[df['Time'] >= baseline_time]
        if subset.empty:
            break

        next_row = subset.iloc[0]

        if next_row['Time'] > last_time:
            new_rows.append(next_row)
            last_time = next_row['Time']

        baseline_time += time_interval

    return pd.DataFrame(new_rows).reset_index(drop=True)


def _load_data(**params):
    verbose = params.get("verbose", True)
    data_directory = params.get("data_directory")
    signal_based_extraction = params.get("signal_based_extraction")

    if verbose:
        print("Loading the datasets...")

    all_files = sorted(
        [os.path.join(data_directory, f)
         for f in os.listdir(data_directory)
         if f.endswith('.csv')]
    )

    df_list = []
    for file_path in all_files:
        df_raw = pd.read_csv(file_path)

        df_raw = df_raw[df_raw["LoadingStarted"] == 1].reset_index(drop=True)

        df = _cartesian_from_encoders(df_raw)

        if signal_based_extraction:
            if verbose:
                print(f"Applying signal-based extraction to: {file_path}")
            df = df.sort_values(by="Time").reset_index(drop=True)
            df = _signal_based_extraction(df, **params)

        df_list.append(df)

    row_counts = [len(dfi) for dfi in df_list]
    return df_list, row_counts
