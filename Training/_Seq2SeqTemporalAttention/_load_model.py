from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

from ._attention_layer import TemporalAttention

def _load_model(**params):
    verbose = params.get("verbose")
    model_name = params.get("model_name")
    if verbose:
        print("loading the pre-trained model...")

    model = load_model(f'./SavedModels/{model_name}/seq2seq_trajectory_model_3d.h5', custom_objects={'mse': MeanSquaredError(), 'Attention': TemporalAttention})
    
    return model