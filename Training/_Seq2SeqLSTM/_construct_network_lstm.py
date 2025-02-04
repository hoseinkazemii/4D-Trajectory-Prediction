import contextlib
from io import StringIO
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def _construct_network(**params):
    """
    Dynamically builds one or more Seq2Seq LSTM models based on the 'coordinates' list.
    For example, if coordinates = ["XZ", "Y"], this function constructs:
      - LSTM_Model_XZ (input shape = (sequence_length, 2), output shape = (prediction_horizon, 2))
      - LSTM_Model_Y (input shape = (sequence_length, 1), output shape = (prediction_horizon, 1))
    Returns a dictionary of { coordinate_string: compiled_keras_model }.
    """
    verbose = params.get("verbose", True)
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    coordinates_list = params.get("coordinates")  # e.g. ["XZ", "Y"] or ["X", "Y", "Z"], etc.
    log = params.get("log")
    coord_to_dim = params.get("coord_to_dim")  # e.g. {"X": 1, "Y": 1, "Z": 1, "XZ": 2, "XYZ": 3, ...}
    learning_rate = params.get("learning_rate")
    decay_steps = params.get("decay_steps")
    decay_rate = params.get("decay_rate")

    if verbose:
        print("Building Seq2Seq LSTM models for coordinates:", coordinates_list)

    models_dict = {}

    for coord_str in coordinates_list:
        if coord_str not in coord_to_dim:
            raise ValueError(f"Unknown coordinate pattern '{coord_str}'. Supported keys: {list(coord_to_dim.keys())}")
        in_dim = coord_to_dim[coord_str]
        out_dim = in_dim  # same number of dimensions for predictions

        # ----- Encoder -----
        # Input shape: (sequence_length, in_dim)
        encoder_inputs = Input(shape=(sequence_length, in_dim), name=f"encoder_input_{coord_str}_lstm")
        # First LSTM layer returns full sequences
        encoder_lstm1 = LSTM(64, return_sequences=True, dropout=0.2, name=f"encoder_lstm1_{coord_str}_lstm")
        # Second LSTM layer returns the last hidden state (and cell state)
        encoder_lstm2 = LSTM(64, return_state=True, dropout=0.2, name=f"encoder_lstm2_{coord_str}_lstm")
        encoder_outputs = encoder_lstm1(encoder_inputs)
        encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs)
        
        # ----- Decoder -----
        # Repeat the encoder’s last hidden state for 'prediction_horizon' time steps
        decoder_inputs = RepeatVector(prediction_horizon, name=f"repeat_vector_{coord_str}_lstm")(state_h)
        # One LSTM layer for decoding
        decoder_lstm = LSTM(64, return_sequences=True, dropout=0.2, name=f"decoder_lstm_{coord_str}_lstm")
        decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
        # TimeDistributed Dense layer to output the desired number of coordinates
        decoder_dense = TimeDistributed(Dense(out_dim), name=f"decoder_dense_{coord_str}_lstm")
        predictions = decoder_dense(decoder_outputs)

        # ----- Compile Model -----
        lr_schedule = ExponentialDecay(learning_rate, decay_steps, decay_rate)
        optimizer = Adam(learning_rate=lr_schedule)
        model = Model(encoder_inputs, predictions, name=f"LSTM_Model_{coord_str}")
        model.compile(optimizer=optimizer, loss=MeanSquaredError())

        # Log the model summary using your logger
        summary_io = StringIO()
        with contextlib.redirect_stdout(summary_io):
            model.summary()
        log.info(f"LSTM Model {coord_str} Summary:\n" + summary_io.getvalue())
        summary_io.close()

        models_dict[coord_str] = model

    return models_dict
