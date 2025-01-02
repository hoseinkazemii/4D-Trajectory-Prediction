from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, TimeDistributed, RepeatVector, Reshape
)
from tensorflow.keras.losses import MeanSquaredError
from ._attention_layer import Attention

import contextlib
from io import StringIO

def _construct_network(**params):
    """
    Builds two separate models:
      - One for Y coordinate predictions
      - One for XZ coordinate predictions

    Uses:
      - 'sequence_length' timesteps in the encoder
      - 'prediction_horizon' timesteps to be predicted by the decoder
      - A simple attention mechanism that uses the final encoder state as the query.

    Returns:
      (model_Y, model_XZ)
    """
    verbose = params.get("verbose")
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    coordinates = params.get("coordinates")
    log = params.get("log")

    if verbose:
        print("Building the models with attention for 'Y' and 'XZ' coordinates separately...")

    # Initialize placeholders for each model
    model_Y = None
    model_XZ = None

    for coordinate in coordinates:
        if coordinate == 'Y':
            # --------------------------------------------------
            #  1) Encoder
            # --------------------------------------------------
            input_shape = (sequence_length, 1)   # e.g. (None, 10, 1)
            output_shape = (prediction_horizon, 1)
            
            encoder_inputs = Input(shape=input_shape, name='encoder_input_y')
            encoder_lstm = LSTM(
                16, return_sequences=True, return_state=True, name='encoder_lstm_y'
            )
            encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
            # encoder_outputs: (batch_size, sequence_length, 16)
            # state_h, state_c: (batch_size, 16)

            # --------------------------------------------------
            #  2) Attention: final h is the "query"
            # --------------------------------------------------
            # Expand dims so that shape is (batch_size, 1, 16)
            query = Reshape((1, 16), name='reshape_query_y')(state_h)

            # We pass [query, encoder_outputs] to the Attention layer
            context_vector = Attention(name='attention_y')([query, encoder_outputs])
            # context_vector: (batch_size, 16)

            # --------------------------------------------------
            #  3) Decoder
            # --------------------------------------------------
            # We replicate that context_vector for 'prediction_horizon' timesteps
            decoder_inputs = RepeatVector(prediction_horizon, name='repeat_context_y')(context_vector)
            # => shape: (batch_size, prediction_horizon, 16)

            decoder_lstm = LSTM(
                16, return_sequences=True, name='decoder_lstm_y'
            )
            # Provide the encoder final state_h, state_c as initial states
            decoder_outputs = decoder_lstm(
                decoder_inputs, initial_state=[state_h, state_c]
            )
            # => shape: (batch_size, prediction_horizon, 16)

            # Final time-distributed dense to map from 16 -> 1
            decoder_dense = TimeDistributed(Dense(output_shape[1]), name='decoder_dense_y')
            predictions = decoder_dense(decoder_outputs)
            # => shape: (batch_size, prediction_horizon, 1)

            # Build and compile
            model_Y = Model(encoder_inputs, predictions, name='Model_Y')
            model_Y.compile(optimizer='adam', loss=MeanSquaredError())

            # Log the summary
            summary_io = StringIO()
            with contextlib.redirect_stdout(summary_io):
                model_Y.summary()
            log.info("Model Y Summary:\n" + summary_io.getvalue())
            summary_io.close()

        elif coordinate == 'XZ':
            # --------------------------------------------------
            #  1) Encoder
            # --------------------------------------------------
            input_shape = (sequence_length, 2)
            output_shape = (prediction_horizon, 2)
            
            encoder_inputs = Input(shape=input_shape, name='encoder_input_xz')
            encoder_lstm = LSTM(
                16, return_sequences=True, return_state=True, name='encoder_lstm_xz'
            )
            encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
            # encoder_outputs: (batch_size, sequence_length, 16)
            # state_h, state_c: (batch_size, 16)

            # --------------------------------------------------
            #  2) Attention
            # --------------------------------------------------
            query = Reshape((1, 16), name='reshape_query_xz')(state_h)
            context_vector = Attention(name='attention_xz')([query, encoder_outputs])
            # => shape: (batch_size, 16)

            # --------------------------------------------------
            #  3) Decoder
            # --------------------------------------------------
            decoder_inputs = RepeatVector(prediction_horizon, name='repeat_context_xz')(context_vector)
            # => shape: (batch_size, prediction_horizon, 16)

            decoder_lstm = LSTM(
                16, return_sequences=True, name='decoder_lstm_xz'
            )
            decoder_outputs = decoder_lstm(
                decoder_inputs, initial_state=[state_h, state_c]
            )
            # => shape: (batch_size, prediction_horizon, 16)

            decoder_dense = TimeDistributed(Dense(output_shape[1]), name='decoder_dense_xz')
            predictions = decoder_dense(decoder_outputs)
            # => shape: (batch_size, prediction_horizon, 2)

            # Build and compile
            model_XZ = Model(encoder_inputs, predictions, name='Model_XZ')
            model_XZ.compile(optimizer='adam', loss=MeanSquaredError())

            # Log the summary
            summary_io = StringIO()
            with contextlib.redirect_stdout(summary_io):
                model_XZ.summary()
            log.info("Model XZ Summary:\n" + summary_io.getvalue())
            summary_io.close()

    # Return whichever models were created
    return model_Y, model_XZ





# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, RepeatVector
# from tensorflow.keras.losses import MeanSquaredError
# from ._attention_layer import Attention

# from io import StringIO
# import contextlib


# def _construct_network(**params):
#     verbose = params.get("verbose")
#     sequence_length = params.get("sequence_length")
#     prediction_horizon = params.get("prediction_horizon")
#     coordinates = params.get("coordinates")
#     log = params.get("log")
#     if verbose:
#         print("building the models with attention for 'Y' and 'XZ' coordinates separately...")

#     for coordinate in coordinates:
#         if coordinate == 'Y':
#             input_shape = (sequence_length, 1)
#             output_shape = (prediction_horizon, 1)
#             encoder_inputs = Input(shape=input_shape)
#             encoder_lstm = LSTM(16, return_sequences=True, return_state=True)
#             encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

#             attention = Attention()([encoder_outputs, encoder_outputs])

#             decoder_lstm = LSTM(16, return_sequences=True, return_state=False)
#             decoder_lstm_inputs = RepeatVector(prediction_horizon)(attention)
#             decoder_outputs = decoder_lstm(decoder_lstm_inputs, initial_state=[state_h, state_c])
            
#             decoder_dense = TimeDistributed(Dense(output_shape[1]))
#             decoder_outputs = decoder_dense(decoder_outputs)

#             model_Y = Model(encoder_inputs, decoder_outputs)
#             model_Y.compile(optimizer='adam', loss=MeanSquaredError())

#             summary_io = StringIO()
#             with contextlib.redirect_stdout(summary_io):
#                 model_Y.summary()
#             summary_str = summary_io.getvalue()
#             summary_io.close()
#             # Log the model summary
#             log.info("Model Y Summary:\n" + summary_str)

#         elif coordinate == 'XZ':
#             input_shape = (sequence_length, 2)
#             output_shape = (prediction_horizon, 2)
#             encoder_inputs = Input(shape=input_shape)
#             encoder_lstm = LSTM(2, return_sequences=True, return_state=True)
#             encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

#             attention = Attention()([encoder_outputs, encoder_outputs])
            
#             decoder_lstm = LSTM(2, return_sequences=True, return_state=False)
#             decoder_lstm_inputs = RepeatVector(prediction_horizon)(attention)
#             decoder_outputs = decoder_lstm(decoder_lstm_inputs, initial_state=[state_h, state_c])
            
#             decoder_dense = TimeDistributed(Dense(output_shape[1]))
#             decoder_outputs = decoder_dense(decoder_outputs)

#             model_XZ = Model(encoder_inputs, decoder_outputs)
#             model_XZ.compile(optimizer='adam', loss=MeanSquaredError())

#             summary_io = StringIO()
#             with contextlib.redirect_stdout(summary_io):
#                 model_XZ.summary()
#             summary_str = summary_io.getvalue()
#             summary_io.close()
#             # Log the model summary
#             log.info("Model XZ Summary:\n" + summary_str)

#     return model_Y, model_XZ