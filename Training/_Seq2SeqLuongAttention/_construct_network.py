# Luong attention WITHOUT layer normalization and residual connection
import contextlib
from io import StringIO

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional , TimeDistributed, RepeatVector, Reshape
)
from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from ._attention_layer import Attention


def _construct_network(**params):
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
        print("Building Seq2Seq models (with Luong Attention) for coordinates: ", coordinates_list)

    # This will store all created models
    models_dict = {}


    for coord_str in coordinates_list:
        # Determine input/output dimension from the coordinate string
        if coord_str not in coord_to_dim:
            raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
                             f"Supported keys: {list(coord_to_dim.keys())}")

        in_dim = coord_to_dim[coord_str]   # e.g. 2 if coord_str == "XZ"
        out_dim = in_dim                  # same dimension for predictions

        # --------------------------------------------------
        #  1) Encoder
        # --------------------------------------------------
        encoder_inputs = Input(shape=(sequence_length, in_dim), 
                               name=f"encoder_input_{coord_str}_mha")

        # Add multiple LSTM layers
        encoder_lstm1 = Bidirectional(LSTM(64, return_sequences=True, name='encoder_1', dropout=0.2, recurrent_dropout=0.2))
        encoder_lstm2 = LSTM(64, return_sequences=True, return_state=True, name='encoder_2')
        encoder_outputs = encoder_lstm1(encoder_inputs)
        encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs)
        # => (batch_size, sequence_length, hidden_dim)

        # --------------------------------------------------
        #  2) Attention: final h is the "query"
        # --------------------------------------------------
        # Expand dims so that shape is (batch_size, 1, hidden_dim)
        query = Reshape((1, 64), name=f"reshape_query_{coord_str}_mha")(state_h)

        # We pass [query, encoder_outputs] to the Attention layer
        context_vector = Attention(name='attention_y')([query, encoder_outputs])
        # context_vector: (batch_size, hidden_dim)

        # --------------------------------------------------
        #  3) Decoder
        # --------------------------------------------------
        # Repeat the 2D context for 'prediction_horizon' steps
        decoder_inputs = RepeatVector(prediction_horizon, 
                                      name=f"repeat_context_{coord_str}_mha")(context_vector)
        # => (batch_size, prediction_horizon, hidden_dim)

        decoder_lstm1 = LSTM(64, return_sequences=True, dropout=0.2, name='decoder_lstm1')
        decoder_lstm2 = LSTM(64, return_sequences=True, dropout=0.2, name='decoder_lstm2')

        decoder_outputs = decoder_lstm1(decoder_inputs, initial_state=[state_h, state_c])
        decoder_outputs = decoder_lstm2(decoder_outputs)
        # => (batch_size, prediction_horizon, hidden_dim)

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

    # Return whichever models were created
    return models_dict



#######################
# Luong attention WITH layer normalization and residual connection

# import contextlib
# from io import StringIO

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input, LSTM, Dense, Bidirectional, TimeDistributed, RepeatVector, 
#     Reshape, LayerNormalization, Add
# )
# from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.schedules import ExponentialDecay

# from ._attention_layer import Attention

# def _construct_network(**params):
#     verbose = params.get("verbose", True)
#     sequence_length = params.get("sequence_length")
#     prediction_horizon = params.get("prediction_horizon")
#     coordinates_list = params.get("coordinates")  # e.g. ["XZ", "Y"] or ["X","Y","Z"]
#     log = params.get("log")
#     coord_to_dim = params.get("coord_to_dim")
#     learning_rate = params.get("learning_rate")
#     decay_steps = params.get("decay_steps")
#     decay_rate = params.get("decay_rate")

#     if verbose:
#         print("Building Seq2Seq models (with Multi-Head Attention) for coordinates: ", coordinates_list)

#     # This will store all created models
#     models_dict = {}

#     for coord_str in coordinates_list:
#         # Determine input/output dimension from the coordinate string
#         if coord_str not in coord_to_dim:
#             raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
#                            f"Supported keys: {list(coord_to_dim.keys())}")

#         in_dim = coord_to_dim[coord_str]   # e.g. 2 if coord_str == "XZ"
#         out_dim = in_dim                   # same dimension for predictions
#         hidden_dim = 64  # Base hidden dimension

#         # --------------------------------------------------
#         #  1) Encoder
#         # --------------------------------------------------
#         encoder_inputs = Input(shape=(sequence_length, in_dim), 
#                              name=f"encoder_input_{coord_str}_mha")
        
#         # First encoder layer with layer normalization
#         encoder_lstm1 = Bidirectional(LSTM(hidden_dim // 2, return_sequences=True,
#                                          dropout=0.2, recurrent_dropout=0.2),
#                                     name='encoder_1')
#         encoder_outputs = encoder_lstm1(encoder_inputs)
#         encoder_outputs = LayerNormalization(name='encoder_norm_1')(encoder_outputs)

#         # Second encoder layer with layer normalization
#         encoder_lstm2 = LSTM(hidden_dim, return_sequences=True, return_state=True,
#                            dropout=0.2, recurrent_dropout=0.2,
#                            name='encoder_2')
#         encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs)
#         encoder_outputs = LayerNormalization(name='encoder_norm_2')(encoder_outputs)

#         # --------------------------------------------------
#         #  2) Attention
#         # --------------------------------------------------
#         # Reshape state_h for attention query
#         query = Reshape((1, hidden_dim), name=f"reshape_query_{coord_str}_mha")(state_h)
        
#         # Apply attention mechanism
#         context_vector = Attention(name=f'attention_{coord_str}')([query, encoder_outputs])

#         # --------------------------------------------------
#         #  3) Decoder
#         # --------------------------------------------------
#         # Prepare decoder inputs
#         decoder_inputs = RepeatVector(prediction_horizon, 
#                                     name=f"repeat_context_{coord_str}_mha")(context_vector)

#         # First decoder layer with residual connection and layer normalization
#         decoder_lstm1 = LSTM(hidden_dim, return_sequences=True, dropout=0.2,
#                            name='decoder_lstm1')
#         decoder_outputs = decoder_lstm1(decoder_inputs, initial_state=[state_h, state_c])
#         decoder_outputs = LayerNormalization(name='decoder_norm_1')(decoder_outputs)
        
#         # Add residual connection if shapes match
#         if decoder_outputs.shape[-1] == decoder_inputs.shape[-1]:
#             decoder_outputs = Add(name='residual_1')([decoder_outputs, decoder_inputs])

#         # Second decoder layer with residual connection and layer normalization
#         decoder_lstm2 = LSTM(hidden_dim, return_sequences=True, dropout=0.2,
#                            name='decoder_lstm2')
#         decoder_outputs2 = decoder_lstm2(decoder_outputs)
#         decoder_outputs2 = LayerNormalization(name='decoder_norm_2')(decoder_outputs2)
        
#         # Add residual connection
#         decoder_outputs = Add(name='residual_2')([decoder_outputs2, decoder_outputs])

#         # Output layer
#         decoder_dense = TimeDistributed(Dense(out_dim), 
#                                       name=f"decoder_dense_{coord_str}_mha")
#         predictions = decoder_dense(decoder_outputs)

#         # Build & compile model
#         lr_schedule = ExponentialDecay(learning_rate, decay_steps, decay_rate)
#         optimizer = Adam(learning_rate=lr_schedule)
        
#         model = Model(encoder_inputs, predictions, 
#                      name=f"MHA_Model_{coord_str}")
#         model.compile(optimizer=optimizer, 
#                      loss=MeanSquaredError(),
#                      metrics=['mae'])  # Added MAE metric

#         # Log model summary
#         summary_io = StringIO()
#         with contextlib.redirect_stdout(summary_io):
#             model.summary()
#         log.info(f"MultiHeadAttention - Model {coord_str} Summary:\n" + summary_io.getvalue())
#         summary_io.close()

#         # Store the model in the dictionary
#         models_dict[coord_str] = model

#     return models_dict