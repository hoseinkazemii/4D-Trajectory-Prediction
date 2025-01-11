import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class DecoderCell(Layer):
    def __init__(self, lstm_units, attention_layer, out_dim, enc_seq_len, enc_hidden_dim, **kwargs):
        super().__init__(**kwargs)
        
        # Create a custom LSTM cell without using the built-in LSTMCell
        # This gives us more control over the internal operations
        self.units = lstm_units
        
        # LSTM weights
        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        
        # Other components
        self.attention_layer = attention_layer
        self.output_dense = Dense(out_dim)
        
        # Dimensions
        self.enc_seq_len = enc_seq_len
        self.enc_hidden_dim = enc_hidden_dim
        
        # Dropout (using fixed rate)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Initialize LSTM weights
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer='glorot_uniform'
        )
        
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer='orthogonal'
        )
        
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            name='bias',
            initializer='zeros'
        )
        
        self.built = True

    @property
    def state_size(self):
        return [
            self.units,  # h
            self.units,  # c
            (self.enc_seq_len, self.enc_hidden_dim)  # encoder_outputs
        ]

    def lstm_step(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous hidden state
        c_tm1 = states[1]  # previous cell state
        
        if training:
            inputs = self.dropout(inputs, training=training)
            h_tm1 = self.dropout(h_tm1, training=training)
        
        # Gates computation
        z = tf.matmul(inputs, self.kernel)
        z += tf.matmul(h_tm1, self.recurrent_kernel)
        z = tf.nn.bias_add(z, self.bias)
        
        z0, z1, z2, z3 = tf.split(z, 4, axis=-1)
        
        i = tf.sigmoid(z0)  # input gate
        f = tf.sigmoid(z1)  # forget gate
        c = f * c_tm1 + i * tf.tanh(z2)
        o = tf.sigmoid(z3)  # output gate
        
        h = o * tf.tanh(c)
        
        return h, [h, c]

    def call(self, inputs, states, training=None):
        h, c, enc_out = states
        
        # 1) LSTM step
        lstm_output, [new_h, new_c] = self.lstm_step(
            inputs, 
            [h, c],
            training=training
        )
        
        # 2) Attention
        context_vector = self.attention_layer(new_h, enc_out)
        
        # 3) Combine context with LSTM output
        combined = tf.concat([lstm_output, context_vector], axis=-1)
        
        # 4) Final output
        output = self.output_dense(combined)
        
        return output, [new_h, new_c, enc_out]

    def get_config(self):
        config = super().get_config()
        config.update({
            'lstm_units': self.units,
            'out_dim': self.output_dense.units,
            'enc_seq_len': self.enc_seq_len,
            'enc_hidden_dim': self.enc_hidden_dim
        })
        return config
