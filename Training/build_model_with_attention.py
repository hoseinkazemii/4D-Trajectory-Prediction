# Training/build_model_with_attention.py
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, RepeatVector
from tensorflow.keras.losses import MeanSquaredError
from .attention_layer import Attention

def _build_model_with_attention(**params):
    verbose = params.get("verbose")
    sequence_length = params.get("sequence_length")
    warmup = params.get("warmup")
    coordinate = params.get("coordinate")

    if warmup:
        if verbose:
            print("loading the pre-trained model...")
        model = load_model('./SavedModels/seq2seq_trajectory_model_3d.h5', custom_objects={'mse': MeanSquaredError(), 'Attention': Attention})
    else:
        if verbose:
            print("building the model with attention...")
        
        if coordinate == 'Y':
            input_shape = (sequence_length, 1)
            output_shape = 1
        else:
            input_shape = (sequence_length, 2)
            output_shape = 2
        
        encoder_inputs = Input(shape=input_shape)
        encoder_lstm = LSTM(16, return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

        attention = Attention()([encoder_outputs, encoder_outputs])
        
        decoder_lstm = LSTM(16, return_sequences=True, return_state=False)
        decoder_lstm_inputs = RepeatVector(sequence_length)(attention)
        decoder_outputs = decoder_lstm(decoder_lstm_inputs, initial_state=[state_h, state_c])
        
        decoder_dense = TimeDistributed(Dense(output_shape))
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model(encoder_inputs, decoder_outputs)
        model.compile(optimizer='adam', loss=MeanSquaredError())
        model.summary()

    return model