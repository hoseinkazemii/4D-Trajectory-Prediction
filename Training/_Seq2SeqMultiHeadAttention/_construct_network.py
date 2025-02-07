import contextlib
from io import StringIO

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional,
    TimeDistributed, Add, LayerNormalization
)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from ._attention_layer import MultiHeadEncDecAttention

def _construct_network(**params):
    """
    Dynamically builds one or more models based on 'coordinates'.
    
    Teacher Forcing version:
    - The decoder has a separate 'decoder_inputs' placeholder
      which should be fed the ground-truth future trajectory
      during training (teacher forcing).
    """
    verbose = params.get("verbose", True)
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    coordinates_list = params.get("coordinates")  # e.g. ["XZ", "Y"] etc.
    log = params.get("log")
    coord_to_dim = params.get("coord_to_dim")
    learning_rate = params.get("learning_rate")
    decay_steps = params.get("decay_steps")
    decay_rate = params.get("decay_rate")

    if verbose:
        print("Building Seq2Seq Teacher-Forcing models for coordinates:", coordinates_list)

    models_dict = {}

    for coord_str in coordinates_list:
        if coord_str not in coord_to_dim:
            raise ValueError(f"Unknown coordinate '{coord_str}'. "
                             f"Supported: {list(coord_to_dim.keys())}")

        in_dim = coord_to_dim[coord_str]
        out_dim = in_dim  # Usually the same dimension for outputs

        # --------------- Encoder ---------------
        encoder_inputs = Input(
            shape=(sequence_length, in_dim),
            name=f"encoder_input_{coord_str}_mha"
        )

        # (A) Bidirectional LSTM 1 with Layer Normalization
        encoder_bidirectional_1 = Bidirectional(
            LSTM(256, return_sequences=True, name='encoder_bidirectional_1')
        )(encoder_inputs)
        encoder_bidirectional_1 = LayerNormalization()(encoder_bidirectional_1)

        # (B) Bidirectional LSTM 2 with Layer Normalization
        encoder_bidirectional_2 = Bidirectional(
            LSTM(256, return_sequences=True, name='encoder_bidirectional_2')
        )(encoder_bidirectional_1)
        encoder_bidirectional_2 = LayerNormalization()(encoder_bidirectional_2)

        # (C) Unidirectional LSTM with Layer Normalization
        encoder_outputs, state_h, state_c = LSTM(
            256, return_sequences=True, return_state=True,
            name='encoder_lstm_3'
        )(encoder_bidirectional_2)
        encoder_outputs = LayerNormalization()(encoder_outputs)

        # --------------- Decoder ---------------
        decoder_inputs = Input(
            shape=(prediction_horizon, out_dim),
            name=f"decoder_input_{coord_str}_mha"
        )

        # (A) LSTM layers with Layer Normalization
        decoder_outputs = LSTM(256, return_sequences=True, name='decoder_lstm1')(decoder_inputs, initial_state=[state_h, state_c])
        decoder_outputs = LayerNormalization()(decoder_outputs)

        decoder_outputs = LSTM(256, return_sequences=True, name='decoder_lstm2')(decoder_outputs)
        decoder_outputs = LayerNormalization()(decoder_outputs)

        decoder_outputs = LSTM(256, return_sequences=True, name='decoder_lstm3')(decoder_outputs)
        decoder_outputs = LayerNormalization()(decoder_outputs)

        decoder_outputs = LSTM(256, return_sequences=True, name='decoder_lstm4')(decoder_outputs)
        decoder_outputs = LayerNormalization()(decoder_outputs)

        # --------------- MultiHeadAttention ---------------
        mha_layer = MultiHeadEncDecAttention(num_heads=8, key_dim=32, name=f"mha_{coord_str}")
        dynamic_context = mha_layer([decoder_outputs, encoder_outputs])

        # Residual connection with Layer Normalization
        decoder_combined_context = Add(name=f"add_context_{coord_str}_mha")(
            [decoder_outputs, dynamic_context]
        )
        decoder_combined_context = LayerNormalization()(decoder_combined_context)



        # --------------- Final Time-Distributed Dense ---------------
        predictions = TimeDistributed(
            Dense(out_dim), name=f"decoder_dense_{coord_str}_mha"
        )(dense_layer_2)

        # --------------- Compile Model ---------------
        lr_schedule = ExponentialDecay(learning_rate, decay_steps, decay_rate)
        optimizer = Adam(learning_rate=lr_schedule)

        model = Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=predictions,
            name=f"MHA_TF_Model_{coord_str}"
        )
        model.compile(optimizer=optimizer, loss=MeanSquaredError())

        summary_io = StringIO()
        with contextlib.redirect_stdout(summary_io):
            model.summary()
        log.info(f"TeacherForcing - Model {coord_str} Summary:\n" + summary_io.getvalue())
        summary_io.close()

        models_dict[coord_str] = model

    return models_dict









import contextlib
from io import StringIO

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional,
    TimeDistributed, Add, LayerNormalization
)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from ._attention_layer import MultiHeadEncDecAttention

def _construct_network(**params):
    """
    Dynamically builds one or more models based on 'coordinates'.
    
    Teacher Forcing version:
    - The decoder has a separate 'decoder_inputs' placeholder
      which should be fed the ground-truth future trajectory
      during training (teacher forcing).
    """
    verbose = params.get("verbose", True)
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    coordinates_list = params.get("coordinates")  # e.g. ["XZ", "Y"] etc.
    log = params.get("log")
    coord_to_dim = params.get("coord_to_dim")
    learning_rate = params.get("learning_rate")
    decay_steps = params.get("decay_steps")
    decay_rate = params.get("decay_rate")

    if verbose:
        print("Building Seq2Seq Teacher-Forcing models for coordinates:", coordinates_list)

    models_dict = {}

    for coord_str in coordinates_list:
        if coord_str not in coord_to_dim:
            raise ValueError(f"Unknown coordinate '{coord_str}'. "
                             f"Supported: {list(coord_to_dim.keys())}")

        in_dim = coord_to_dim[coord_str]
        out_dim = in_dim  # Usually the same dimension for outputs

        # --------------- Encoder ---------------
        encoder_inputs = Input(
            shape=(sequence_length, in_dim),
            name=f"encoder_input_{coord_str}_mha"
        )

        # (A) Bidirectional LSTM 1
        encoder_bidirectional_1 = Bidirectional(
            LSTM(256, return_sequences=True, name='encoder_bidirectional_1')
        )(encoder_inputs)

        # (B) Bidirectional LSTM 2
        encoder_bidirectional_2 = Bidirectional(
            LSTM(256, return_sequences=True, name='encoder_bidirectional_2')
        )(encoder_bidirectional_1)

        # (C) Unidirectional LSTM, returns final state
        encoder_outputs, state_h, state_c = LSTM(
            256, return_sequences=True, return_state=True,
            name='encoder_lstm_3'
        )(encoder_bidirectional_2)

        # --------------- Decoder ---------------
        # Instead of RepeatVector, we feed ground-truth sequences
        decoder_inputs = Input(
            shape=(prediction_horizon, out_dim),
            name=f"decoder_input_{coord_str}_mha"
        )
        
        decoder_lstm1 = LSTM(256, return_sequences=True, name='decoder_lstm1')
        decoder_lstm2 = LSTM(256, return_sequences=True, name='decoder_lstm2')
        decoder_lstm3 = LSTM(256, return_sequences=True, name='decoder_lstm3')
        # decoder_lstm4 = LSTM(256, return_sequences=True, name='decoder_lstm4')

        # Pass decoder_inputs through the LSTMs with initial state from the encoder
        decoder_outputs = decoder_lstm1(decoder_inputs, initial_state=[state_h, state_c])
        decoder_outputs = decoder_lstm2(decoder_outputs)
        decoder_outputs = decoder_lstm3(decoder_outputs)
        # decoder_outputs = decoder_lstm4(decoder_outputs)

        # --------------- MultiHeadAttention ---------------
        mha_layer = MultiHeadEncDecAttention(num_heads=8, key_dim=32, name=f"mha_{coord_str}")
        dynamic_context = mha_layer([decoder_outputs, encoder_outputs])

        # Residual connection
        decoder_combined_context = Add(name=f"add_context_{coord_str}_mha")(
            [decoder_outputs, dynamic_context]
        )
        decoder_combined_context = LayerNormalization()(decoder_combined_context)

        # --------------- Final Time-Distributed Dense ---------------
        decoder_dense = TimeDistributed(
            Dense(out_dim), name=f"decoder_dense_{coord_str}_mha"
        )
        predictions = decoder_dense(decoder_combined_context)

        # --------------- Compile Model ---------------
        lr_schedule = ExponentialDecay(learning_rate, decay_steps, decay_rate)
        optimizer = Adam(learning_rate=lr_schedule)

        model = Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=predictions,
            name=f"MHA_TF_Model_{coord_str}"
        )
        model.compile(optimizer=optimizer, loss=MeanSquaredError())

        summary_io = StringIO()
        with contextlib.redirect_stdout(summary_io):
            model.summary()
        log.info(f"TeacherForcing - Model {coord_str} Summary:\n" + summary_io.getvalue())
        summary_io.close()

        models_dict[coord_str] = model

    return models_dict