#### Multiheaded attention WITHOUT residual connection:
import contextlib
from io import StringIO

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional , TimeDistributed, RepeatVector, Reshape
)
from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
    learning_rate = params.get("learning_rate")
    decay_steps = params.get("decay_steps")
    decay_rate = params.get("decay_rate")

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
        encoder_lstm1 = Bidirectional(LSTM(64, return_sequences=True, name='encoder_1', dropout=0.2, recurrent_dropout=0.2))
        encoder_lstm2 = LSTM(64, return_sequences=True, return_state=True, name='encoder_2')
        encoder_outputs = encoder_lstm1(encoder_inputs)
        encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs)
        # => (batch_size, sequence_length, 64)

        # Multi-head cross-attention
        # We treat state_h => (batch_size, 64) as the "query" 
        # and reshape it to (batch_size, 1, 64).
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

        decoder_lstm1 = LSTM(64, return_sequences=True, dropout=0.2, name='decoder_lstm1')
        decoder_lstm2 = LSTM(64, return_sequences=True, dropout=0.2, name='decoder_lstm2')

        decoder_outputs = decoder_lstm1(decoder_inputs, initial_state=[state_h, state_c])
        decoder_outputs = decoder_lstm2(decoder_outputs)
        # => (batch_size, prediction_horizon, 64)

        decoder_dense = TimeDistributed(Dense(out_dim), 
                                        name=f"decoder_dense_{coord_str}_mha")
        predictions = decoder_dense(decoder_outputs)
        # => (batch_size, prediction_horizon, out_dim)

        # Build & compile the model
        # Use learning rate scheduling
        lr_schedule = ExponentialDecay(
            learning_rate, decay_steps, decay_rate)
        optimizer = Adam(learning_rate=lr_schedule)
        model = Model(encoder_inputs, predictions, 
                      name=f"MHA_Model_{coord_str}")
        model.compile(optimizer=optimizer, loss=MeanSquaredError())

        # # Add early stopping and model checkpointing
        # early_stopping = EarlyStopping(monitor='val_loss', 
        #                             patience=10,
        #                             restore_best_weights=True)
        # model_checkpoint = ModelCheckpoint('best_model.keras', 
        #                                 monitor='val_loss',
        #                                 save_best_only=True)

        # Log model summary
        summary_io = StringIO()
        with contextlib.redirect_stdout(summary_io):
            model.summary()
        log.info(f"MultiHeadAttention - Model {coord_str} Summary:\n" + summary_io.getvalue())
        summary_io.close()

        # Store the model in the dictionary
        models_dict[coord_str] = model

    return models_dict




#### Multiheaded attention WITH residual connection:
# import contextlib
# from io import StringIO

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input, LSTM, Dense, TimeDistributed, RepeatVector, Reshape, Add
# )
# from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.schedules import ExponentialDecay
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# from ._attention_layer import MultiHeadEncDecAttention

# def _construct_network(**params):
#     """
#     Dynamically builds one or more models based on the 'coordinates' list, 
#     with residual (skip) connections in the encoder and decoder LSTMs.
#     """
#     verbose = params.get("verbose", True)
#     sequence_length = params.get("sequence_length")
#     prediction_horizon = params.get("prediction_horizon")
#     coordinates_list = params.get("coordinates")  # e.g. ["XZ", "Y"] or ["X","Y","Z"], etc.
#     log = params.get("log")
#     coord_to_dim = params.get("coord_to_dim")
#     learning_rate = params.get("learning_rate", 1e-3)
#     decay_steps = params.get("decay_steps", 1000)
#     decay_rate = params.get("decay_rate", 0.9)

#     if verbose:
#         print("Building Seq2Seq models (with Multi-Head Attention + Residual LSTMs) for coordinates: ", coordinates_list)

#     # This will store all created models
#     models_dict = {}

#     for coord_str in coordinates_list:
#         # Determine input/output dimension from the coordinate string
#         if coord_str not in coord_to_dim:
#             raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
#                              f"Supported keys: {list(coord_to_dim.keys())}")

#         in_dim = coord_to_dim[coord_str]   # e.g. 2 if coord_str == "XZ"
#         out_dim = in_dim                  # same dimension for predictions

#         # -------------------------------------------------
#         #  Encoder
#         # -------------------------------------------------
#         # Input shape => (sequence_length, in_dim)
#         encoder_inputs = Input(
#             shape=(sequence_length, in_dim),
#             name=f"encoder_input_{coord_str}_mha"
#         )

#         # LSTM layer 1 (returns full sequences)
#         encoder_lstm1 = LSTM(
#             64, return_sequences=True, name=f"encoder_1_{coord_str}",
#             dropout=0.2, recurrent_dropout=0.2
#         )
#         # LSTM layer 2 (returns sequences + final states)
#         encoder_lstm2 = LSTM(
#             64, return_sequences=True, return_state=True,
#             name=f"encoder_2_{coord_str}"
#         )

#         # Pass data through the first LSTM
#         enc_x1 = encoder_lstm1(encoder_inputs)  
#         # => shape (batch, seq_len, 64)

#         # Pass enc_x1 to second LSTM
#         enc_x2_out, state_h, state_c = encoder_lstm2(enc_x1)
#         # => enc_x2_out shape: (batch, seq_len, 64), plus state_h, state_c

#         # Residual connection => add the outputs from both LSTMs
#         # They both have shape (batch, seq_len, 64), so we can do a direct Add
#         enc_x2 = Add(name=f"encoder_residual_{coord_str}")([enc_x2_out, enc_x1])
#         # => shape (batch, seq_len, 64)

#         # -------------------------------------------------
#         #  Attention
#         # -------------------------------------------------
#         # We'll treat the final LSTM's state_h => (batch, 64) as "query"
#         # Reshape => (batch, 1, 64)
#         query = Reshape((1, 64), name=f"reshape_query_{coord_str}_mha")(state_h)

#         # Multi-head attention uses enc_x2 (the residual result) as "value"/"key"
#         mha_layer = MultiHeadEncDecAttention(num_heads=8, key_dim=32, name=f"mha_{coord_str}")
#         context_vector = mha_layer([query, enc_x2])  # => (batch, 1, 64)

#         # Flatten context => (batch, 64)
#         context_vector_2d = Reshape((64,), name=f"reshape_context_{coord_str}_mha")(context_vector)

#         # -------------------------------------------------
#         #  Decoder
#         # -------------------------------------------------
#         # We repeat context_vector_2d for 'prediction_horizon' steps
#         decoder_inputs = RepeatVector(
#             prediction_horizon,
#             name=f"repeat_context_{coord_str}_mha"
#         )(context_vector_2d)
#         # => shape (batch, prediction_horizon, 64)

#         # Two LSTM layers in the decoder
#         decoder_lstm1 = LSTM(64, return_sequences=True, dropout=0.2,
#                              name=f"decoder_lstm1_{coord_str}")
#         decoder_lstm2 = LSTM(64, return_sequences=True, dropout=0.2,
#                              name=f"decoder_lstm2_{coord_str}")

#         dec_x1 = decoder_lstm1(decoder_inputs, initial_state=[state_h, state_c])
#         dec_x2 = decoder_lstm2(dec_x1)

#         # Residual connection in the decoder => Add the outputs of the 2 LSTM layers
#         # Both dec_x1, dec_x2 => shape (batch, prediction_horizon, 64)
#         dec_x2_res = Add(name=f"decoder_residual_{coord_str}")([dec_x2, dec_x1])

#         # Final TimeDistributed Dense
#         decoder_dense = TimeDistributed(
#             Dense(out_dim), name=f"decoder_dense_{coord_str}_mha"
#         )
#         predictions = decoder_dense(dec_x2_res)
#         # => (batch, prediction_horizon, out_dim)

#         # -------------------------------------------------
#         #  Compile Model
#         # -------------------------------------------------
#         lr_schedule = ExponentialDecay(learning_rate, decay_steps, decay_rate)
#         optimizer = Adam(learning_rate=lr_schedule)

#         model = Model(encoder_inputs, predictions, name=f"MHA_Model_{coord_str}")
#         model.compile(optimizer=optimizer, loss=MeanSquaredError())

#         # Optional: EarlyStopping, ModelCheckpoint as needed:
#         # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         # model_checkpoint = ModelCheckpoint(f"best_model_{coord_str}.h5", save_best_only=True)

#         # Log model summary
#         summary_io = StringIO()
#         with contextlib.redirect_stdout(summary_io):
#             model.summary()
#         log.info(f"MultiHeadAttention + Residual - Model {coord_str} Summary:\n" + summary_io.getvalue())
#         summary_io.close()

#         # Store the model in the dictionary
#         models_dict[coord_str] = model

#     return models_dict
