import contextlib
from io import StringIO

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, TimeDistributed, RepeatVector
)
from tensorflow.keras.losses import MeanSquaredError

# Import the TemporalAttention layer
from ._attention_layer import TemporalAttention

def _construct_network(**params):
    """
    Builds two models, one for 'Y' and one for 'XZ' coordinates, 
    using 'TemporalAttention'.
    """
    verbose = params.get("verbose", True)
    sequence_length = params["sequence_length"]
    prediction_horizon = params["prediction_horizon"]
    coordinates = params["coordinates"]  # a list like ["Y", "XZ"]
    log = params["log"]

    # We'll return these models at the end
    model_Y = None
    model_XZ = None

    if verbose:
        print("Building models (Temporal Attention) for 'Y' and/or 'XZ'...")

    for coordinate in coordinates:
        if coordinate == "Y":
            # ---------------------------------------
            #  1) Encoder
            # ---------------------------------------
            encoder_inputs = Input(shape=(sequence_length, 1), name="encoder_input_y")
            encoder_lstm = LSTM(16, return_sequences=True, return_state=True, name="encoder_lstm_y")
            encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
            # encoder_outputs => (batch_size, sequence_length, 16)

            # ---------------------------------------
            #  2) Temporal Attention 
            # ---------------------------------------
            attention_layer = TemporalAttention(name="temporal_attention_y")
            context_vector = attention_layer(encoder_outputs)
            # context_vector => (batch_size, 16)

            # ---------------------------------------
            #  3) Decoder
            # ---------------------------------------
            decoder_inputs = RepeatVector(prediction_horizon, name="repeat_context_y")(context_vector)
            # => (batch_size, prediction_horizon, 16)

            decoder_lstm = LSTM(16, return_sequences=True, name="decoder_lstm_y")
            decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
            # => (batch_size, prediction_horizon, 16)

            decoder_dense = TimeDistributed(Dense(1), name="decoder_dense_y")
            predictions = decoder_dense(decoder_outputs)
            # => (batch_size, prediction_horizon, 1)

            # Build & compile
            model_Y = Model(encoder_inputs, predictions, name="TemporalAttention_Y")
            model_Y.compile(optimizer="adam", loss=MeanSquaredError())

            # Optional logging
            summary_io = StringIO()
            with contextlib.redirect_stdout(summary_io):
                model_Y.summary()
            log.info("TemporalAttention - Model Y Summary:\n" + summary_io.getvalue())
            summary_io.close()

        elif coordinate == "XZ":
            # ---------------------------------------
            #  1) Encoder
            # ---------------------------------------
            encoder_inputs = Input(shape=(sequence_length, 2), name="encoder_input_xz")
            encoder_lstm = LSTM(16, return_sequences=True, return_state=True, name="encoder_lstm_xz")
            encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
            # encoder_outputs => (batch_size, sequence_length, 16)

            # ---------------------------------------
            #  2) Temporal Attention
            # ---------------------------------------
            attention_layer = TemporalAttention(name="temporal_attention_xz")
            context_vector = attention_layer(encoder_outputs)
            # context_vector => (batch_size, 16)

            # ---------------------------------------
            #  3) Decoder
            # ---------------------------------------
            decoder_inputs = RepeatVector(prediction_horizon, name="repeat_context_xz")(context_vector)
            # => (batch_size, prediction_horizon, 16)

            decoder_lstm = LSTM(16, return_sequences=True, name="decoder_lstm_xz")
            decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
            # => (batch_size, prediction_horizon, 16)

            decoder_dense = TimeDistributed(Dense(2), name="decoder_dense_xz")
            predictions = decoder_dense(decoder_outputs)
            # => (batch_size, prediction_horizon, 2)

            # Build & compile
            model_XZ = Model(encoder_inputs, predictions, name="TemporalAttention_XZ")
            model_XZ.compile(optimizer="adam", loss=MeanSquaredError())

            # Optional logging
            summary_io = StringIO()
            with contextlib.redirect_stdout(summary_io):
                model_XZ.summary()
            log.info("TemporalAttention - Model XZ Summary:\n" + summary_io.getvalue())
            summary_io.close()

    return model_Y, model_XZ