import contextlib
from io import StringIO

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, TimeDistributed, RepeatVector
)
from tensorflow.keras.optimizers import Adam

from ._attention_layer import TemporalAttention
from ._physics_loss import piml_loss_factory

def _construct_network(**params):
    """
    Builds three separate models: one for X, one for Y, one for Z,
    each with a temporal attention mechanism + physics-informed time derivative constraints.
    """
    verbose = params.get("verbose", True)
    sequence_length = params["sequence_length"]
    prediction_horizon = params["prediction_horizon"]
    log = params["log"]

    # Hyperparams for the PI loss
    alpha = params.get("pi_alpha", 0.1)   # weight for velocity penalty
    beta = params.get("pi_beta", 0.1)    # weight for acceleration penalty
    v_max = params.get("v_max", 1.0)     # max velocity
    a_max = params.get("a_max", 0.5)     # max acceleration
    dt = params.get("dt", 1.0)           # time delta for consecutive predictions

    # Create the custom PIML loss
    piml_loss = piml_loss_factory(alpha=alpha, beta=beta, v_max=v_max, a_max=a_max, dt=dt)

    if verbose:
        print("Building three models (X, Y, Z) with Temporal Attention + Time-derivative PIML Loss...")

    # We'll store the final compiled models here
    model_X = None
    model_Y = None
    model_Z = None

    # -------------
    # Model for X
    # -------------
    encoder_inputs_x = Input(shape=(sequence_length, 1), name="encoder_input_x")
    encoder_lstm_x = LSTM(16, return_sequences=True, return_state=True, name="encoder_lstm_x")
    encoder_outputs_x, state_h_x, state_c_x = encoder_lstm_x(encoder_inputs_x)

    attention_layer_x = TemporalAttention(name="temporal_attention_x")
    context_vector_x = attention_layer_x(encoder_outputs_x)

    decoder_inputs_x = RepeatVector(prediction_horizon, name="repeat_context_x")(context_vector_x)
    decoder_lstm_x = LSTM(16, return_sequences=True, name="decoder_lstm_x")
    decoder_outputs_x = decoder_lstm_x(decoder_inputs_x, initial_state=[state_h_x, state_c_x])

    decoder_dense_x = TimeDistributed(Dense(1), name="decoder_dense_x")
    predictions_x = decoder_dense_x(decoder_outputs_x)

    model_X = Model(encoder_inputs_x, predictions_x, name="TemporalAttention_PIML_X")
    model_X.compile(optimizer=Adam(), loss=piml_loss)

    summary_io = StringIO()
    with contextlib.redirect_stdout(summary_io):
        model_X.summary()
    log.info("Model X Summary (PIML):\n" + summary_io.getvalue())
    summary_io.close()

    # -------------
    # Model for Y
    # -------------
    encoder_inputs_y = Input(shape=(sequence_length, 1), name="encoder_input_y")
    encoder_lstm_y = LSTM(16, return_sequences=True, return_state=True, name="encoder_lstm_y")
    encoder_outputs_y, state_h_y, state_c_y = encoder_lstm_y(encoder_inputs_y)

    attention_layer_y = TemporalAttention(name="temporal_attention_y")
    context_vector_y = attention_layer_y(encoder_outputs_y)

    decoder_inputs_y = RepeatVector(prediction_horizon, name="repeat_context_y")(context_vector_y)
    decoder_lstm_y = LSTM(16, return_sequences=True, name="decoder_lstm_y")
    decoder_outputs_y = decoder_lstm_y(decoder_inputs_y, initial_state=[state_h_y, state_c_y])

    decoder_dense_y = TimeDistributed(Dense(1), name="decoder_dense_y")
    predictions_y = decoder_dense_y(decoder_outputs_y)

    model_Y = Model(encoder_inputs_y, predictions_y, name="TemporalAttention_PIML_Y")
    model_Y.compile(optimizer=Adam(), loss=piml_loss)

    summary_io = StringIO()
    with contextlib.redirect_stdout(summary_io):
        model_Y.summary()
    log.info("Model Y Summary (PIML):\n" + summary_io.getvalue())
    summary_io.close()

    # -------------
    # Model for Z
    # -------------
    encoder_inputs_z = Input(shape=(sequence_length, 1), name="encoder_input_z")
    encoder_lstm_z = LSTM(16, return_sequences=True, return_state=True, name="encoder_lstm_z")
    encoder_outputs_z, state_h_z, state_c_z = encoder_lstm_z(encoder_inputs_z)

    attention_layer_z = TemporalAttention(name="temporal_attention_z")
    context_vector_z = attention_layer_z(encoder_outputs_z)

    decoder_inputs_z = RepeatVector(prediction_horizon, name="repeat_context_z")(context_vector_z)
    decoder_lstm_z = LSTM(16, return_sequences=True, name="decoder_lstm_z")
    decoder_outputs_z = decoder_lstm_z(decoder_inputs_z, initial_state=[state_h_z, state_c_z])

    decoder_dense_z = TimeDistributed(Dense(1), name="decoder_dense_z")
    predictions_z = decoder_dense_z(decoder_outputs_z)

    model_Z = Model(encoder_inputs_z, predictions_z, name="TemporalAttention_PIML_Z")
    model_Z.compile(optimizer=Adam(), loss=piml_loss)

    summary_io = StringIO()
    with contextlib.redirect_stdout(summary_io):
        model_Z.summary()
    log.info("Model Z Summary (PIML):\n" + summary_io.getvalue())
    summary_io.close()

    return model_X, model_Y, model_Z