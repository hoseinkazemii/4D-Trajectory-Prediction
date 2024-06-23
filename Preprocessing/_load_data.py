import pandas as pd


def _load_data(**params):
    verbose = params.get("verbose")
    data_directory = params.get("data_directory")
    if verbose:
        print("loading the data...")

    # Load the dataset
    df = pd.read_csv(data_directory + 'sample_dataset.csv')

    return df