import contextlib
from io import StringIO

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, TimeDistributed, RepeatVector, Reshape
)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from ._attention_layer import MultiHeadEncDecAttention


def _construct_network(**params):
    """
    Dynamically builds one or more models based on the 'coordinates' list.

    For example, if coordinates = ["XZ", "Y"], this function constructs:
      - MHA_Model_XZ (input shape = (seq_len, 2), output shape = (pred_horizon, 2))
      - MHA_Model_Y  (input shape = (seq_len, 1), output shape = (pred_horizon, 1))

    Returns a dictionary of { coordinate_string: compiled_keras_model }.
    """
    verbose = params.get("verbose", True)
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    coordinates_list = params.get("coordinates") # e.g. ["XZ", "Y"] or ["X","Y","Z"], etc.
    log = params.get("log")
    coord_to_dim = params.get("coord_to_dim")

    if verbose:
        print("Building Seq2Seq models (with Multi-Head Attention) for coordinates: ", coordinates_list)

    # This will store all created models
    models_dict = {}

    for coord_str in coordinates_list:
        # Determine input/output dimension from the coordinate string
        if coord_str not in coord_to_dim:
            raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
                             f"Supported keys: {list(coord_to_dim.keys())}")

        in_dim = coord_to_dim[coord_str]   # e.g. 2 if coord_str == "XZ"
        out_dim = in_dim                  # same dimension for predictions

        # Encoder
        # Use shape (sequence_length, in_dim), e.g. (seq_len, 2) for XZ
        encoder_inputs = Input(shape=(sequence_length, in_dim), 
                               name=f"encoder_input_{coord_str}_mha")

        # Add multiple LSTM layers
        encoder_lstm1 = LSTM(64, return_sequences=True, name='encoder_1', dropout=0.2, recurrent_dropout=0.2)
        encoder_lstm2 = LSTM(64, return_sequences=True, return_state=True, name='encoder_2')
        encoder_outputs = encoder_lstm1(encoder_inputs)
        encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs)
        # => (batch_size, sequence_length, 16)      

        # Multi-head cross-attention
        # We treat state_h => (batch_size, 16) as the "query" 
        # and reshape it to (batch_size, 1, 16).
        query = Reshape((1, 64), name=f"reshape_query_{coord_str}_mha")(state_h)

        mha_layer = MultiHeadEncDecAttention(num_heads=8, key_dim=32, 
                                             name=f"mha_{coord_str}")
        context_vector = mha_layer([query, encoder_outputs])  
        # => shape (batch_size, 1, 64)

        # Optionally flatten context to 2D
        context_vector_2d = Reshape((64,), 
                                    name=f"reshape_context_{coord_str}_mha")(context_vector)

        # Decoder
        # Repeat the 2D context for 'prediction_horizon' steps
        decoder_inputs = RepeatVector(prediction_horizon, 
                                      name=f"repeat_context_{coord_str}_mha")(context_vector_2d)
        # => (batch_size, prediction_horizon, 64)

        decoder_lstm = LSTM(64, return_sequences=True, dropout=0.2,
                            name=f"decoder_lstm_{coord_str}_mha")
        decoder_outputs = decoder_lstm(decoder_inputs, 
                                       initial_state=[state_h, state_c])
        # => (batch_size, prediction_horizon, 64)

        decoder_dense = TimeDistributed(Dense(out_dim), 
                                        name=f"decoder_dense_{coord_str}_mha")
        predictions = decoder_dense(decoder_outputs)
        # => (batch_size, prediction_horizon, out_dim)

        # Build & compile the model
        model = Model(encoder_inputs, predictions, 
                      name=f"MHA_Model_{coord_str}")
        model.compile(optimizer=Adam(), loss=MeanSquaredError())

        # Log model summary
        summary_io = StringIO()
        with contextlib.redirect_stdout(summary_io):
            model.summary()
        log.info(f"MultiHeadAttention - Model {coord_str} Summary:\n" + summary_io.getvalue())
        summary_io.close()

        # Store the model in the dictionary
        models_dict[coord_str] = model

    return models_dict
