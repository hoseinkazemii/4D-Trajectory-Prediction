from ._GATTemporalModel import GATTemporalModel

def _construct_model(**params):
    device = params.get('device')
    in_channels = params.get('in_channels')
    prediction_horizon = params.get('prediction_horizon')
    verbose = params.get('verbose', True)

    if verbose:
        print(f"Constructing GATTemporalModel")

    model = GATTemporalModel(
        in_channels=in_channels,
        prediction_horizon=prediction_horizon
    ).to(device)

    return model
