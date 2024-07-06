from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, RepeatVector
from tensorflow.keras.losses import MeanSquaredError
from ._attention_layer import Attention

from io import StringIO
import contextlib


def _construct_network(**params):
    verbose = params.get("verbose")
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    coordinates = params.get("coordinates")
    log = params.get("log")
    if verbose:
        print("building the models with attention for 'Y' and 'XZ' coordinates separately...")

    for coordinate in coordinates:
        if coordinate == 'Y':
            input_shape = (sequence_length, 1)
            output_shape = (prediction_horizon, 1)
            encoder_inputs = Input(shape=input_shape)
            encoder_lstm = LSTM(16, return_sequences=True, return_state=True)
            encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

            attention = Attention()([encoder_outputs, encoder_outputs])
            
            decoder_lstm = LSTM(16, return_sequences=True, return_state=False)
            decoder_lstm_inputs = RepeatVector(prediction_horizon)(attention)
            decoder_outputs = decoder_lstm(decoder_lstm_inputs, initial_state=[state_h, state_c])
            
            decoder_dense = TimeDistributed(Dense(output_shape[1]))
            decoder_outputs = decoder_dense(decoder_outputs)

            model_Y = Model(encoder_inputs, decoder_outputs)
            model_Y.compile(optimizer='adam', loss=MeanSquaredError())

            summary_io = StringIO()
            with contextlib.redirect_stdout(summary_io):
                model_Y.summary()
            summary_str = summary_io.getvalue()
            summary_io.close()
            # Log the model summary
            log.info("Model Y Summary:\n" + summary_str)

        elif coordinate == 'XZ':
            input_shape = (sequence_length, 2)
            output_shape = (prediction_horizon, 2)
            encoder_inputs = Input(shape=input_shape)
            encoder_lstm = LSTM(16, return_sequences=True, return_state=True)
            encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

            attention = Attention()([encoder_outputs, encoder_outputs])
            
            decoder_lstm = LSTM(16, return_sequences=True, return_state=False)
            decoder_lstm_inputs = RepeatVector(prediction_horizon)(attention)
            decoder_outputs = decoder_lstm(decoder_lstm_inputs, initial_state=[state_h, state_c])
            
            decoder_dense = TimeDistributed(Dense(output_shape[1]))
            decoder_outputs = decoder_dense(decoder_outputs)

            model_XZ = Model(encoder_inputs, decoder_outputs)
            model_XZ.compile(optimizer='adam', loss=MeanSquaredError())

            summary_io = StringIO()
            with contextlib.redirect_stdout(summary_io):
                model_XZ.summary()
            summary_str = summary_io.getvalue()
            summary_io.close()
            # Log the model summary
            log.info("Model XZ Summary:\n" + summary_str)

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
#     coordinates = params.get("coordinates")
#     log = params.get("log")
#     if verbose:
#         print("building the models with attention for 'Y' and 'XZ' coordinates separately...")
    
#     for coordinate in coordinates:
#         if coordinate == 'Y':
#             input_shape = (sequence_length, 1)
#             output_shape = 1
#             encoder_inputs = Input(shape=input_shape)
#             encoder_lstm = LSTM(16, return_sequences=True, return_state=True)
#             encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

#             attention = Attention()([encoder_outputs, encoder_outputs])
            
#             decoder_lstm = LSTM(16, return_sequences=True, return_state=False)
#             decoder_lstm_inputs = RepeatVector(sequence_length)(attention)
#             decoder_outputs = decoder_lstm(decoder_lstm_inputs, initial_state=[state_h, state_c])
            
#             decoder_dense = TimeDistributed(Dense(output_shape))
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
#             output_shape = 2
#             encoder_inputs = Input(shape=input_shape)
#             encoder_lstm = LSTM(16, return_sequences=True, return_state=True)
#             encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

#             attention = Attention()([encoder_outputs, encoder_outputs])
            
#             decoder_lstm = LSTM(16, return_sequences=True, return_state=False)
#             decoder_lstm_inputs = RepeatVector(sequence_length)(attention)
#             decoder_outputs = decoder_lstm(decoder_lstm_inputs, initial_state=[state_h, state_c])
            
#             decoder_dense = TimeDistributed(Dense(output_shape))
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