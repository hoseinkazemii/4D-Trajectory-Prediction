import contextlib
from io import StringIO
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, TimeDistributed, Conv2D, Lambda, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def _construct_network_convlstm(**params):
    """
    For each coordinate group (e.g. "XYZ", "XZ", "Y", etc.) builds a ConvLSTM-based
    encoder-decoder model.

    The model expects inputs of shape (sequence_length, 1, in_dim, 1), where in_dim is
    the number of coordinates in the group (e.g. 3 for "XYZ" or 2 for "XZ"). Internally,
    the network uses two ConvLSTM2D layers in the encoder; then the encoder output is
    repeated for prediction_horizon timesteps and passed through two additional
    ConvLSTM2D layers and a TimeDistributed Conv2D to yield predictions of shape
    (prediction_horizon, in_dim).
    """
    verbose = params.get("verbose", True)
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    coordinates_list = params.get("coordinates")  # e.g. ["XYZ", "XZ", "Y"], etc.
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
        in_dim = coord_to_dim[coord_str]  # e.g. 3 if coord_str == "XYZ"
        # The output dimension is the same (prediction of the same coordinate components)
        out_dim = in_dim

        # Reshape the original time-series (batch, sequence_length, in_dim)
        # to (batch, sequence_length, 1, in_dim, 1) so that ConvLSTM2D works.
        encoder_inputs = Input(shape=(sequence_length, 1, in_dim, 1),
                               name=f"encoder_input_{coord_str}_convlstm")

        # Encoder: two ConvLSTM2D layers.
        encoder = ConvLSTM2D(
            filters=64, kernel_size=(1, 1), padding='same',
            return_sequences=True, name=f"encoder_convlstm1_{coord_str}"
        )(encoder_inputs)
        encoder_output = ConvLSTM2D(
            filters=64, kernel_size=(1, 1), padding='same',
            return_sequences=False, name=f"encoder_convlstm2_{coord_str}"
        )(encoder)

        # Repeat the encoder output for each timestep in the prediction horizon.
        # The encoder_output has shape (batch, 1, in_dim, 64) and we want to create
        # a tensor of shape (batch, prediction_horizon, 1, in_dim, 64).
        def repeat_fn(x):
            # Expand dims to add a new time dimension (axis=1)
            x_expanded = tf.expand_dims(x, axis=1)  # Now shape: (batch, 1, 1, in_dim, 64)
            # Tile along the new axis to repeat the encoder output 'prediction_horizon' times.
            return tf.tile(x_expanded, [1, prediction_horizon, 1, 1, 1])

        # Provide output_shape so Keras can infer the shape correctly.
        decoder_inputs = Lambda(
            repeat_fn,
            output_shape=lambda input_shape: (input_shape[0], prediction_horizon, input_shape[1], input_shape[2], input_shape[3]),
            name=f"repeat_encoder_output_{coord_str}_convlstm"
        )(encoder_output)

        # Decoder: two ConvLSTM2D layers.
        decoder = ConvLSTM2D(
            filters=64, kernel_size=(1, 1), padding='same',
            return_sequences=True, name=f"decoder_convlstm1_{coord_str}"
        )(decoder_inputs)
        decoder = ConvLSTM2D(
            filters=64, kernel_size=(1, 1), padding='same',
            return_sequences=True, name=f"decoder_convlstm2_{coord_str}"
        )(decoder)

        # A TimeDistributed Conv2D layer to map the 64 filters back to 1 channel.
        # This yields a tensor of shape (batch, prediction_horizon, 1, in_dim, 1).
        decoder_output = TimeDistributed(
            Conv2D(filters=1, kernel_size=(1, 1), activation='linear', padding='same'),
            name=f"decoder_conv2d_{coord_str}"
        )(decoder)

        # Finally, reshape to (batch, prediction_horizon, in_dim) to match your targets.
        predictions = Reshape((prediction_horizon, out_dim),
                              name=f"reshape_output_{coord_str}_convlstm")(decoder_output)

        # Compile the model with an exponentially decaying learning rate.
        lr_schedule = ExponentialDecay(learning_rate, decay_steps, decay_rate)
        optimizer = Adam(learning_rate=lr_schedule)
        model = Model(encoder_inputs, predictions,
                      name=f"ConvLSTM_Model_{coord_str}")
        model.compile(optimizer=optimizer, loss=MeanSquaredError())

        # Log the model summary.
        summary_io = StringIO()
        with contextlib.redirect_stdout(summary_io):
            model.summary()
        log.info(f"ConvLSTM - Model {coord_str} Summary:\n" + summary_io.getvalue())
        summary_io.close()

        models_dict[coord_str] = model

    return models_dict
