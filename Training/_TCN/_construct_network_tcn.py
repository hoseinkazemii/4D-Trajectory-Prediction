import contextlib
from io import StringIO
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, TimeDistributed, Lambda
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from ._tcn_layer import TCNLayer

def _construct_network_tcn(**params):
    coordinates_list = params.get("coordinates")
    coord_to_dim = params.get("coord_to_dim")
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    verbose = params.get("verbose", True)
    log = params.get("log")
    learning_rate = params.get("learning_rate")
    decay_steps = params.get("decay_steps")
    decay_rate = params.get("decay_rate")
    run_eagerly = params.get("run_eagerly")

    if verbose:
        print("Constructing TCN models for coordinates:", coordinates_list)
        print(f"Prediction horizon: {prediction_horizon}")

    models_dict = {}

    for coord_str in coordinates_list:
        in_dim = coord_to_dim[coord_str]
        out_dim = in_dim

        inputs = Input(shape=(sequence_length, in_dim), name=f"input_{coord_str}")

        tcn_output = TCNLayer(
            nb_filters=128,
            kernel_size=3,
            nb_stacks=2,
            dilations=[1, 2, 4],
            padding='causal',
            use_skip_connections=True,
            dropout_rate=0.2,
            return_sequences=True,
            name=f"tcn_{coord_str}"
        )(inputs)

        last_timestep_features = Lambda(
            lambda t: t[:, -1, :],
            name=f"extract_last_timestep_{coord_str}"
        )(tcn_output)

        repeated_features = Lambda(
            lambda x: tf.repeat(tf.expand_dims(x, axis=1), repeats=prediction_horizon, axis=1),
            name=f"repeat_features_{coord_str}"
        )(last_timestep_features)

        decoded_features = TimeDistributed(
            Dense(64, activation='relu'),
            name=f"decode_dense1_{coord_str}"
        )(repeated_features)

        decoded_features = TimeDistributed(Dense(128, activation='relu'))(repeated_features)
        decoded_features = TimeDistributed(Dense(64, activation='relu'))(decoded_features)
        outputs = TimeDistributed(Dense(out_dim))(decoded_features)

        model = Model(inputs, outputs, name=f"TCNModel_{coord_str}")
        lr_schedule = ExponentialDecay(learning_rate, decay_steps, decay_rate)
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss=Huber(delta=1.0),
            run_eagerly=run_eagerly
        )

        summary_io = StringIO()
        with contextlib.redirect_stdout(summary_io):
            model.summary()
        if log:
            log.info(f"TCN Model for {coord_str}:\n{summary_io.getvalue()}")
        else:
            print(summary_io.getvalue())

        models_dict[coord_str] = model

    return models_dict
