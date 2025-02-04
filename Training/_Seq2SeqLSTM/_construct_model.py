from ._construct_network_lstm import _construct_network
from ._load_model import _load_model  # Assumes you already have this helper

def _construct_model(**params):
    warm_up = params.get('warm_up')
    log = params.get("log")
    if warm_up:
        try:
            model = _load_model(**params)
            log.info("\n\n------------\nA trained LSTM model is loaded\n------------\n\n")
            return model
        except OSError:
            print("The model is not trained before. No saved models found")
    # Otherwise, build the network from scratch
    models_dict = _construct_network(**params)
    return models_dict
