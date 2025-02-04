import contextlib
from io import StringIO
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import MeanSquaredError

def _construct_network(**params):
    """
    Dynamically builds one or more GRU-based seq2seq models based on the 'coordinates' list.
    For example, if coordinates = ["XZ", "Y"], then for "XZ" the input shape is (sequence_length, 2)
    and the output shape is (prediction_horizon, 2).
    """
    verbose = params.get("verbose", True)
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    coordinates_list = params.get("coordinates")  # e.g. ["XZ", "Y"] or ["X", "Y", "Z"]
    log = params.get("log")
    coord_to_dim = params.get("coord_to_dim")  # e.g. { "X": 1, "Y": 1, "Z": 1, "XZ": 2, "XYZ": 3, ... }
    learning_rate = params.get("learning_rate")
    decay_steps = params.get("decay_steps")
    decay_rate = params.get("decay_rate")
    
    if verbose:
        print("Building Seq2Seq models (with GRU) for coordinates:", coordinates_list)
    
    models_dict = {}
    for coord_str in coordinates_list:
        if coord_str not in coord_to_dim:
            raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
                             f"Supported keys: {list(coord_to_dim.keys())}")
        in_dim = coord_to_dim[coord_str]  # e.g. 2 if coord_str == "XZ"
        out_dim = in_dim  # same dimension for prediction

        # === Encoder ===
        # Input shape: (sequence_length, in_dim)
        encoder_inputs = Input(shape=(sequence_length, in_dim),
                               name=f"encoder_input_{coord_str}_gru")
        # Use a two‐layer GRU encoder (the first GRU returns sequences)
        encoder_gru1 = GRU(64, return_sequences=True, dropout=0.2,
                           name=f"encoder_gru1_{coord_str}")
        encoder_gru2 = GRU(64, return_state=True, dropout=0.2,
                           name=f"encoder_gru2_{coord_str}")
        encoder_output = encoder_gru1(encoder_inputs)
        # encoder_gru2 returns (output, state) but we only need the final state
        _, state = encoder_gru2(encoder_output)  # state shape: (batch_size, 64)

        # === Decoder ===
        # Use the encoder’s final state as the context for each prediction step.
        # Repeat the state (now shape: (batch_size, 64)) prediction_horizon times.
        decoder_inputs = RepeatVector(prediction_horizon,
                                      name=f"repeat_context_{coord_str}_gru")(state)
        # Two GRU layers in the decoder
        decoder_gru1 = GRU(64, return_sequences=True, dropout=0.2,
                           name=f"decoder_gru1_{coord_str}")
        decoder_gru2 = GRU(64, return_sequences=True, dropout=0.2,
                           name=f"decoder_gru2_{coord_str}")
        # You can optionally initialize the first decoder GRU with the encoder state.
        decoder_output = decoder_gru1(decoder_inputs, initial_state=state)
        decoder_output = decoder_gru2(decoder_output)

        # Final Dense layer to produce the coordinate outputs at each time step
        decoder_dense = TimeDistributed(Dense(out_dim),
                                        name=f"decoder_dense_{coord_str}_gru")
        predictions = decoder_dense(decoder_output)

        # === Build and Compile the Model ===
        lr_schedule = ExponentialDecay(learning_rate, decay_steps, decay_rate)
        optimizer = Adam(learning_rate=lr_schedule)
        model = Model(encoder_inputs, predictions, name=f"GRU_Model_{coord_str}")
        model.compile(optimizer=optimizer, loss=MeanSquaredError())

        # Log the model summary to the provided logger
        summary_io = StringIO()
        with contextlib.redirect_stdout(summary_io):
            model.summary()
        log.info(f"GRU Model {coord_str} Summary:\n" + summary_io.getvalue())
        summary_io.close()

        # Save the model in the dictionary
        models_dict[coord_str] = model
    return models_dict
