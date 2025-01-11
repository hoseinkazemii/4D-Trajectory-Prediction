import contextlib
from io import StringIO

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from ._attention_layer import TemporalAttention
from ._decoder_cell import DecoderCell

def _construct_network(**params):
    verbose = params.get("verbose", True)
    sequence_length = params["sequence_length"]
    prediction_horizon = params["prediction_horizon"]
    coordinates_list = params["coordinates"]
    coord_to_dim = params["coord_to_dim"]
    learning_rate = params["learning_rate"]
    decay_steps = params["decay_steps"]
    decay_rate = params["decay_rate"]
    run_eagerly = params["run_eagerly"]
    log = params.get("log")

    # Configure mixed precision if needed
    mixed_precision = params.get("mixed_precision", False)
    if mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    if verbose:
        print(f"Building Seq2Seq + Temporal Attention for coords: {coordinates_list}")

    models_dict = {}

    for coord_str in coordinates_list:
        if coord_str not in coord_to_dim:
            raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
                           f"Supported keys: {list(coord_to_dim.keys())}")

        in_dim = coord_to_dim[coord_str]
        out_dim = in_dim

        # ----- 1) Encoder -----
        encoder_inputs = Input(
            shape=(sequence_length, in_dim),
            name=f"encoder_input_{coord_str}"
        )

        # Create encoder layers with explicit names
        encoder_lstm1 = Bidirectional(
            LSTM(64, return_sequences=True, 
                 dropout=0.0,
                 recurrent_dropout=0.0,  # Explicitly set recurrent_dropout
                 name=f'lstm1_{coord_str}'),
            name=f'bidirectional_{coord_str}'
        )
        
        encoder_lstm2 = LSTM(
            64, 
            return_sequences=True,
            return_state=True,
            dropout=0.0,
            recurrent_dropout=0.0,  # Explicitly set recurrent_dropout
            name=f'lstm2_{coord_str}'
        )

        # Apply encoder layers
        x = encoder_lstm1(encoder_inputs)
        encoder_outputs, state_h, state_c = encoder_lstm2(x)

        # ----- 2) Attention & Decoder -----
        attention_layer = TemporalAttention(
            hidden_dim=64,
            name=f"temporal_attention_{coord_str}"
        )

        decoder_cell = DecoderCell(
            lstm_units=64,
            attention_layer=attention_layer,
            out_dim=out_dim,
            enc_seq_len=sequence_length,
            enc_hidden_dim=64,
            name=f"decoder_cell_{coord_str}"
        )

        decoder_inputs = Input(
            shape=(prediction_horizon, in_dim),
            name=f"decoder_input_{coord_str}"
        )

        # Create RNN layer with explicit configuration
        decoder_rnn = tf.keras.layers.RNN(
            decoder_cell,
            return_sequences=True,
            stateful=False,  # Explicitly set stateful
            name=f"decoder_rnn_{coord_str}"
        )

        # Initial states including encoder outputs
        initial_states = [state_h, state_c, encoder_outputs]

        # Apply decoder
        decoder_outputs = decoder_rnn(
            decoder_inputs,
            initial_state=initial_states,
            training=True  # Explicitly set training flag
        )

        # ----- 3) Model -----
        model = Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=decoder_outputs,
            name=f"TemporalAttentionModel_{coord_str}"
        )

        # ----- 4) Compile -----
        lr_schedule = ExponentialDecay(
            learning_rate,
            decay_steps,
            decay_rate,
            staircase=False
        )
        
        optimizer = Adam(learning_rate=lr_schedule)
        
        # Use run_eagerly for debugging if needed
        model.compile(
            optimizer=optimizer,
            loss=MeanSquaredError(),
            run_eagerly= run_eagerly
        )

        # Optional: Log model summary
        summary_io = StringIO()
        with contextlib.redirect_stdout(summary_io):
            model.summary()
        if log is not None:
            log.info(f"MultiPass TemporalAttention - Model {coord_str} Summary:\n" 
                     + summary_io.getvalue())
        summary_io.close()


        models_dict[coord_str] = model

    return models_dict





######################
## Single-pass temporal attention
# import contextlib
# from io import StringIO

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input, LSTM, Dense, Bidirectional, LayerNormalization, TimeDistributed, RepeatVector, Reshape
# )
# from tensorflow.keras.losses import MeanSquaredError, Huber
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.schedules import ExponentialDecay
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# from ._attention_layer import TemporalAttention
# from ._physics_loss import piml_loss_factory

# def _construct_network(**params):
#     """
#     Builds separate models for each coord_str in coordinates list in ,
#     each with a temporal attention mechanism + physics-informed time derivative constraints.
#     """
#     verbose = params.get("verbose", True)
#     sequence_length = params.get("sequence_length")
#     prediction_horizon = params.get("prediction_horizon")
#     coordinates_list = params.get("coordinates") # e.g. ["XZ", "Y"] or ["X","Y","Z"], etc.
#     log = params.get("log")
#     coord_to_dim = params.get("coord_to_dim")
#     learning_rate = params.get("learning_rate")
#     decay_steps = params.get("decay_steps")
#     decay_rate = params.get("decay_rate")

#     if verbose:
#         print("Building Seq2Seq models (with Temporal Attention) for coordinates: ", coordinates_list)

#     # Hyperparams for the PI loss
#     alpha = params.get("pi_alpha", 0.1)   # weight for velocity penalty
#     beta = params.get("pi_beta", 0.1)    # weight for acceleration penalty
#     v_max = params.get("v_max", 1.0)     # max velocity
#     a_max = params.get("a_max", 0.5)     # max acceleration
#     dt = params.get("dt", 1.0)           # time delta for consecutive predictions

#     # Create the custom PIML loss
#     piml_loss = piml_loss_factory(alpha=alpha, beta=beta, v_max=v_max, a_max=a_max, dt=dt)

#     models_dict = {}

#     for coord_str in coordinates_list:
#         # Determine input/output dimension from the coordinate string
#         if coord_str not in coord_to_dim:
#             raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
#                              f"Supported keys: {list(coord_to_dim.keys())}")

#         in_dim = coord_to_dim[coord_str]   # e.g. 2 if coord_str == "XZ"
#         out_dim = in_dim                  # same dimension for predictions

#         # Encoder
#         # Use shape (sequence_length, in_dim), e.g. (seq_len, 2) for XZ
#         encoder_inputs = Input(shape=(sequence_length, in_dim), 
#                                name=f"encoder_input_{coord_str}_temporalattention")

#         # Add multiple LSTM layers
#         encoder_lstm1 = Bidirectional(LSTM(64, return_sequences=True, name='encoder_1', dropout=0.2, recurrent_dropout=0.2))
#         encoder_lstm2 = LSTM(64, return_sequences=True, return_state=True, name='encoder_2')
#         encoder_outputs = encoder_lstm1(encoder_inputs)
#         encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs)
#         # => (batch_size, sequence_length, 64)
#         # encoder_outputs = LayerNormalization()(encoder_outputs)

#         attention_layer = TemporalAttention(name="temporal_attention")
#         context_vector = attention_layer(encoder_outputs)

#         # Decoder
#         # Repeat the 2D context for 'prediction_horizon' steps
#         decoder_inputs = RepeatVector(prediction_horizon, 
#                                       name=f"repeat_context_{coord_str}_temporalattention")(context_vector)
#         # => (batch_size, prediction_horizon, 64)

#         decoder_lstm1 = LSTM(64, return_sequences=True, dropout=0.2, name='decoder_lstm1')
#         decoder_lstm2 = LSTM(64, return_sequences=True, dropout=0.2, name='decoder_lstm2')

#         decoder_outputs = decoder_lstm1(decoder_inputs, initial_state=[state_h, state_c])
#         decoder_outputs = decoder_lstm2(decoder_outputs)
#         # => (batch_size, prediction_horizon, 64)
#         # decoder_outputs = LayerNormalization()(decoder_outputs)

#         decoder_dense = TimeDistributed(Dense(out_dim), 
#                                         name=f"decoder_dense_{coord_str}_temporalattention")
#         predictions = decoder_dense(decoder_outputs)
#         # => (batch_size, prediction_horizon, out_dim)

#         # Build & compile the model
#         # Use learning rate scheduling
#         lr_schedule = ExponentialDecay(
#             learning_rate, decay_steps, decay_rate)
#         optimizer = Adam(learning_rate=lr_schedule)
#         model = Model(encoder_inputs, predictions, 
#                       name=f"TemporalAttention_Model_{coord_str}")
#         model.compile(optimizer=optimizer, loss=MeanSquaredError())
#         # model.compile(optimizer=optimizer, loss=piml_loss)

#         # # Add early stopping and model checkpointing
#         # early_stopping = EarlyStopping(monitor='val_loss', 
#         #                             patience=10,
#         #                             restore_best_weights=True)
#         # model_checkpoint = ModelCheckpoint('best_model.keras', 
#         #                                 monitor='val_loss',
#         #                                 save_best_only=True)

#         # Log model summary
#         summary_io = StringIO()
#         with contextlib.redirect_stdout(summary_io):
#             model.summary()
#         log.info(f"TemporalAttention - Model {coord_str} Summary:\n" + summary_io.getvalue())
#         summary_io.close()

#         # Store the model in the dictionary
#         models_dict[coord_str] = model

#     return models_dict