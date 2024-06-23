from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector

def _build_model(**params):
    verbose = params.get("verbose")
    sequence_length = params.get("sequence_length")
    warmup = params.get("warmup")

    if warmup:
        if verbose:
            print("loading the pre-trained model...")
        model = load_model('seq2seq_trajectory_model_3d.h5', custom_objects={'mse': MeanSquaredError()})
    else:
        if verbose:
            print("building the model...")
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(sequence_length, 3), return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=False))
        model.add(RepeatVector(sequence_length))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(3)))
        model.compile(optimizer='adam', loss=MeanSquaredError())
        model.summary()
    return model