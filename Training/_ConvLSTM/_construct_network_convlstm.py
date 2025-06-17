import contextlib
from io import StringIO
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, TimeDistributed, Conv2D, Lambda, Reshape, Add, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import regularizers

def _construct_network_convlstm(**params):
    verbose = params.get("verbose", True)
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    coordinates_list = params.get("coordinates")
    log = params.get("log")
    coord_to_dim = params.get("coord_to_dim")
    learning_rate = params.get("learning_rate")
    decay_steps = params.get("decay_steps")
    decay_rate = params.get("decay_rate")

    if verbose:
        print("Building ConvLSTM models for coordinates:", coordinates_list)

    models_dict = {}

    for coord_str in coordinates_list:
        if coord_str not in coord_to_dim:
            raise ValueError(f"Unknown coordinate pattern '{coord_str}'. "
                             f"Supported keys: {list(coord_to_dim.keys())}")
        in_dim = coord_to_dim[coord_str]
        out_dim = in_dim

        encoder_inputs = Input(shape=(sequence_length, 1, in_dim, 1),
                               name=f"encoder_input_{coord_str}_convlstm")

        encoder_output1 = ConvLSTM2D(filters=64, kernel_size=(1, in_dim), padding='same',
                                    return_sequences=True, name=f"encoder_convlstm1_{coord_str}")(encoder_inputs)

        encoder_output2 = ConvLSTM2D(filters=64, kernel_size=(1, in_dim), padding='same',
                                    return_sequences=True, name=f"encoder_convlstm2_{coord_str}")(encoder_output1)

        encoder_output = Add()([encoder_output1, encoder_output2])
        encoder_output = ConvLSTM2D(filters=64, kernel_size=(1, in_dim), padding='same',
                                return_sequences=False, name=f"encoder_convlstm3_{coord_str}")(encoder_output)
        encoder_output = Dropout(0.2)(encoder_output)

        def repeat_fn(x):
            x_expanded = tf.expand_dims(x, axis=1)
            return tf.tile(x_expanded, [1, prediction_horizon, 1, 1, 1])

        decoder_inputs = Lambda(
            repeat_fn,
            output_shape=lambda input_shape: (input_shape[0], prediction_horizon, input_shape[1], input_shape[2], input_shape[3]),
            name=f"repeat_encoder_output_{coord_str}_convlstm"
        )(encoder_output)

        decoder_output1 = ConvLSTM2D(
            filters=64, kernel_size=(1, in_dim), padding='same',
            return_sequences=True, name=f"decoder_convlstm1_{coord_str}"
        )(decoder_inputs)
        decoder_output2 = ConvLSTM2D(
            filters=64, kernel_size=(1, in_dim), padding='same',
            return_sequences=True, name=f"decoder_convlstm2_{coord_str}"
        )(decoder_output1)
        decoder_output = Add()([decoder_output1, decoder_output2])

        decoder_output = TimeDistributed(
            Conv2D(filters=1, kernel_size=(1, in_dim), activation='linear', padding='same'),
            name=f"decoder_conv2d_{coord_str}"
        )(decoder_output)

        predictions = Reshape((prediction_horizon, out_dim),
                              name=f"reshape_output_{coord_str}_convlstm")(decoder_output)

        lr_schedule = ExponentialDecay(learning_rate, decay_steps, decay_rate)
        optimizer = Adam(learning_rate=lr_schedule)
        model = Model(encoder_inputs, predictions,
                      name=f"ConvLSTM_Model_{coord_str}")
        model.compile(optimizer=optimizer, loss=Huber(delta=1.0))

        summary_io = StringIO()
        with contextlib.redirect_stdout(summary_io):
            model.summary()
        log.info(f"ConvLSTM - Model {coord_str} Summary:\n" + summary_io.getvalue())
        summary_io.close()

        models_dict[coord_str] = model

    return models_dict