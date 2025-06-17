import contextlib
from io import StringIO

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer, Input, LSTM, Dense, Add, Dropout, RepeatVector, TimeDistributed
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from ._attention_layer import MultiHeadEncDecAttention

def _construct_network(**params):
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
        print("Constructing Seq2Seq RepeatVector models (Global Attention) for coordinates:", coordinates_list)

    models_dict = {}

    for coord_str in coordinates_list:
        in_dim = coord_to_dim[coord_str]
        out_dim = in_dim

        encoder_inputs = Input(
            shape=(sequence_length, in_dim),
            name=f"encoder_input_{coord_str}"
        )

        # LSTM #1
        enc_out1 = LSTM(128, return_sequences=True, name=f"encoder_lstm1_{coord_str}")(encoder_inputs)
        enc_out1 = Dropout(0.2)(enc_out1)

        # LSTM #2
        enc_out2 = LSTM(128, return_sequences=True, name=f"encoder_lstm2_{coord_str}")(enc_out1)
        enc_out2 = Dropout(0.2)(enc_out2)

        # Encoder residual
        enc_out2_res = Add(name=f"encoder_residual1_{coord_str}")([enc_out1, enc_out2])

        # LSTM #3: returns final sequence + final states
        enc_out3, state_h, state_c = LSTM(
            128, return_sequences=True, return_state=True,
            name=f"encoder_lstm3_{coord_str}"
        )(enc_out2_res)

        repeated_context = RepeatVector(prediction_horizon, name=f"repeat_{coord_str}")(state_h)

        dec_out1 = LSTM(128, return_sequences=True, name=f"decoder_lstm1_{coord_str}")(
            repeated_context, initial_state=[state_h, state_c]
        )
        dec_out1 = Dropout(0.2)(dec_out1)

        dec_out2 = LSTM(128, return_sequences=True, name=f"decoder_lstm2_{coord_str}")(dec_out1)
        dec_out2 = Dropout(0.2)(dec_out2)

        dec_out2_res = Add(name=f"decoder_residual_{coord_str}")([dec_out1, dec_out2])

        attention_layer = MultiHeadEncDecAttention(
            num_heads=4, 
            key_dim=8, 
            ff_dim=128, 
            dropout=0.2, 
            name=f"mha_{coord_str}"
        )
        cross_out = attention_layer([dec_out2_res, enc_out3])

        decoder_dense = TimeDistributed(Dense(out_dim), name=f"dense_out_{coord_str}")
        outputs = decoder_dense(cross_out)

        model = Model(encoder_inputs, outputs, name=f"RepeatVecModel_{coord_str}")

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
            log.info(f"GlobalAtt RepeatVector Model for {coord_str}:\n{summary_io.getvalue()}")
        else:
            print(summary_io.getvalue())

        models_dict[coord_str] = model

    return models_dict
