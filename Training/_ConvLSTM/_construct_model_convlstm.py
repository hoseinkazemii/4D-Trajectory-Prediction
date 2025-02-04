from ._load_model import _load_model
from ._construct_network_convlstm import _construct_network_convlstm

def _construct_model_convlstm(**params):
    warm_up = params.get('warm_up')
    log = params.get("log")
    
    if warm_up:
        try:
            model = _load_model(**params)
            log.info("\n\n------------\nA trained ConvLSTM model is loaded\n------------\n\n")
            return model
        except OSError:
            print("The ConvLSTM model is not trained before. No saved models found")
    
    # Build a new network if not warm‑starting from a saved model
    models_dict = _construct_network_convlstm(**params)
    return models_dict
