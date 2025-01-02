import contextlib
from io import StringIO

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, TimeDistributed, RepeatVector, Reshape
)
from tensorflow.keras.losses import MeanSquaredError

# Import the MultiHeadEncDecAttention from the same ._attention_layer
from ._attention_layer import MultiHeadEncDecAttention

def _construct_network(**params):
    """
    Builds two separate models (Y and XZ) using Keras's MultiHeadAttention 
    in a Seq2Seq fashion.
    """
    verbose = params.get("verbose", True)
    sequence_length = params["sequence_length"]
    prediction_horizon = params["prediction_horizon"]
    coordinates = params["coordinates"]  # e.g. ["Y", "XZ"]
    log = params["log"]

    model_Y = None
    model_XZ = None

    if verbose:
        print("Building models (Multi-Head Attention) for 'Y' and/or 'XZ'...")

    for coordinate in coordinates:
        if coordinate == 'Y':
            # --------------------------------------------------
            #  1) Encoder
            # --------------------------------------------------
            encoder_inputs = Input(shape=(sequence_length, 1), name='encoder_input_y_mha')
            encoder_lstm = LSTM(16, return_sequences=True, return_state=True, name='encoder_lstm_y_mha')
            encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
            # encoder_outputs => (batch_size, sequence_length, 16)

            # --------------------------------------------------
            #  2) Multi-head cross-attention
            #     We take state_h => (batch_size, 16), reshape => (batch_size, 1, 16) as "query"
            # --------------------------------------------------
            query = Reshape((1, 16), name='reshape_query_y_mha')(state_h)
            mha_layer = MultiHeadEncDecAttention(num_heads=4, key_dim=16, name='mha_y')
            context_vector = mha_layer([query, encoder_outputs])  # => (batch_size, 1, 16)

            # Optionally flatten context to 2D if you want
            # (batch_size, 16), then RepeatVector
            context_vector_2d = Reshape((16,), name='reshape_context_y_mha')(context_vector)

            # --------------------------------------------------
            #  3) Decoder
            # --------------------------------------------------
            decoder_inputs = RepeatVector(prediction_horizon, name='repeat_context_y_mha')(context_vector_2d)
            # => (batch_size, prediction_horizon, 16)

            decoder_lstm = LSTM(16, return_sequences=True, name='decoder_lstm_y_mha')
            decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
            # => (batch_size, prediction_horizon, 16)

            decoder_dense = TimeDistributed(Dense(1), name='decoder_dense_y_mha')
            predictions = decoder_dense(decoder_outputs)
            # => (batch_size, prediction_horizon, 1)

            model_Y = Model(encoder_inputs, predictions, name='MHA_Model_Y')
            model_Y.compile(optimizer='adam', loss=MeanSquaredError())

            # Logging / summary
            summary_io = StringIO()
            with contextlib.redirect_stdout(summary_io):
                model_Y.summary()
            log.info("MultiHeadAttention - Model Y Summary:\n" + summary_io.getvalue())
            summary_io.close()

        elif coordinate == 'XZ':
            # --------------------------------------------------
            #  1) Encoder
            # --------------------------------------------------
            encoder_inputs = Input(shape=(sequence_length, 2), name='encoder_input_xz_mha')
            encoder_lstm = LSTM(16, return_sequences=True, return_state=True, name='encoder_lstm_xz_mha')
            encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
            # encoder_outputs => (batch_size, sequence_length, 16)

            # --------------------------------------------------
            #  2) Multi-head cross-attention
            # --------------------------------------------------
            query = Reshape((1, 16), name='reshape_query_xz_mha')(state_h)
            mha_layer = MultiHeadEncDecAttention(num_heads=4, key_dim=16, name='mha_xz')
            context_vector = mha_layer([query, encoder_outputs])  # => (batch_size, 1, 16)

            context_vector_2d = Reshape((16,), name='reshape_context_xz_mha')(context_vector)

            # --------------------------------------------------
            #  3) Decoder
            # --------------------------------------------------
            decoder_inputs = RepeatVector(prediction_horizon, name='repeat_context_xz_mha')(context_vector_2d)
            # => (batch_size, prediction_horizon, 16)

            decoder_lstm = LSTM(16, return_sequences=True, name='decoder_lstm_xz_mha')
            decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
            # => (batch_size, prediction_horizon, 16)

            decoder_dense = TimeDistributed(Dense(2), name='decoder_dense_xz_mha')
            predictions = decoder_dense(decoder_outputs)
            # => (batch_size, prediction_horizon, 2)

            model_XZ = Model(encoder_inputs, predictions, name='MHA_Model_XZ')
            model_XZ.compile(optimizer='adam', loss=MeanSquaredError())

            # Logging / summary
            summary_io = StringIO()
            with contextlib.redirect_stdout(summary_io):
                model_XZ.summary()
            log.info("MultiHeadAttention - Model XZ Summary:\n" + summary_io.getvalue())
            summary_io.close()

    return model_Y, model_XZ