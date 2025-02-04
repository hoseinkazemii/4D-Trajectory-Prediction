from ._construct_network_gru import _construct_network
from ._load_model import _load_model  # assumes you have this function defined

def _construct_model(**params):
    warm_up = params.get('warm_up')
    log = params.get("log")
    if warm_up:
        try:
            # Try to load a pre-trained GRU model
            model = _load_model(**params)
            log.info("\n\n------------\nA trained GRU model is loaded\n------------\n\n")
            return model
        except OSError:
            print("The GRU model is not trained before. No saved models found")
    # Otherwise, build new models
    models_dict = _construct_network(**params)
    return models_dict