from tensorflow.keras.models import load_model


def _load_model(**params):
    verbose = params.get("verbose")
    model_name = params.get("model_name")
    if verbose:
        print("loading the pre-trained model...")

    model = load_model(f'./SavedModels/{model_name}/seq2seq_trajectory_model_3d.h5')
    
    return model