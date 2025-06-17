import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class TemporalEncDecAttention(Layer):
    def __init__(self, units, hidden_dim, num_heads, **kwargs):
        super(TemporalEncDecAttention, self).__init__(**kwargs)
        self.units = units
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        assert units % num_heads == 0, "units must be divisible by num_heads"
        self.head_dim = units // num_heads
        
        self.Wq = Dense(units, name="Wq")
        self.Wk = Dense(units, name="Wk")
        self.Wv = Dense(units, name="Wv")

        self.temperature = tf.Variable(tf.sqrt(tf.cast(self.head_dim, tf.float32)), 
                                       trainable=True, name="temperature")
        
        self.output_dense = Dense(hidden_dim, name="output_projection")
        
        self.attention_weights = None

    def get_sinusoidal_position_encoding(self, positions, d_model):
        seq_len = tf.shape(positions)[1]
        
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        
        sin_encoding = tf.sin(position * div_term)
        cos_encoding = tf.cos(position * div_term)
        
        position_encoding = tf.concat([sin_encoding, cos_encoding], axis=-1)
        if d_model % 2 != 0:
            position_encoding = tf.concat([position_encoding, tf.zeros([seq_len, 1])], axis=-1)
        
        position_encoding = tf.expand_dims(position_encoding, 0)
        
        return position_encoding

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def combine_heads(self, x, batch_size):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.units))

    def call(self, inputs, training=None):
        query, value = inputs
        batch_size = tf.shape(query)[0]
        dec_time = tf.shape(query)[1]
        enc_time = tf.shape(value)[1]
        
        dec_positions = tf.range(dec_time, dtype=tf.float32)
        dec_positions = tf.expand_dims(dec_positions, 0)
        dec_positions = tf.tile(dec_positions, [batch_size, 1])
        enc_positions = tf.range(enc_time, dtype=tf.float32)
        enc_positions = tf.expand_dims(enc_positions, 0)
        enc_positions = tf.tile(enc_positions, [batch_size, 1])
        enc_positions = tf.range(enc_time, dtype=tf.float32)
        enc_positions = tf.expand_dims(enc_positions, 0)
        enc_positions = tf.tile(enc_positions, [batch_size, 1])
        
        dec_pos_encoding = self.get_sinusoidal_position_encoding(dec_positions, self.hidden_dim)
        enc_pos_encoding = self.get_sinusoidal_position_encoding(enc_positions, self.hidden_dim)
        
        query = query + dec_pos_encoding
        value = value + enc_pos_encoding

        Q = self.Wq(query)
        K = self.Wk(value)
        V = self.Wv(value)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scores = tf.matmul(Q, K, transpose_b=True)
        scores = scores / self.temperature

        attention_weights = tf.nn.softmax(scores, axis=-1)

        context = tf.matmul(attention_weights, V)

        context = self.combine_heads(context, batch_size)

        context = self.output_dense(context)

        self.attention_weights = tf.reduce_mean(attention_weights, axis=1)
        
        return context

    def get_attention_weights(self):
        return self.attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
        })
        return config
