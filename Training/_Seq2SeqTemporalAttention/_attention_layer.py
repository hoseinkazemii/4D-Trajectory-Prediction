######################
## Multi-pass temporal attention
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class TemporalAttention(Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.W_enc = None
        self.W_dec = None
        self.V = None
        self.attention_weights = None

    def build(self, input_shape):
        # Initialize layers in build() instead of __init__
        self.W_enc = Dense(self.hidden_dim, use_bias=False, name='W_enc')
        self.W_dec = Dense(self.hidden_dim, use_bias=False, name='W_dec')
        self.V = Dense(1, use_bias=False, name='V')
        super().build(input_shape)

    @tf.function
    def call(self, query, value, training=None):
        """
        Args:
            query: (batch_size, hidden_dim) - decoder hidden state
            value: (batch_size, enc_seq_len, hidden_dim) - encoder outputs
            training: bool - whether in training mode
        Returns:
            context_vector: (batch_size, hidden_dim)
        """
        # Ensure inputs are proper tensors
        query = tf.convert_to_tensor(query)
        value = tf.convert_to_tensor(value)
        
        # Add time dimension to query
        query_expanded = tf.expand_dims(query, 1)  # (batch, 1, hidden_dim)
        
        # Transform value and query
        value_transformed = self.W_enc(value)  # (batch, enc_seq_len, hidden_dim)
        query_transformed = self.W_dec(query_expanded)  # (batch, 1, hidden_dim)
        
        # Compute alignment scores
        score = tf.nn.tanh(value_transformed + query_transformed)  # (batch, enc_seq_len, hidden_dim)
        score = self.V(score)  # (batch, enc_seq_len, 1)
        
        # Compute attention weights
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, enc_seq_len, 1)
        
        # Store weights for visualization (without gradient tracking)
        self.attention_weights = tf.stop_gradient(tf.squeeze(attention_weights, -1))
        
        # Compute context vector
        context_vector = tf.reduce_sum(attention_weights * value, axis=1)  # (batch, hidden_dim)
        
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim
        })
        return config



## Single-pass temporal attention
# import tensorflow as tf
# from tensorflow.keras.layers import Layer, Dense

# class TemporalAttention(Layer):
#     """
#     Learns a per-timestep weight distribution (attention) from the encoder outputs.
#     No explicit 'query' is used. This is a simpler 'temporal-only' attention.
#     """
#     def __init__(self, **kwargs):
#         super(TemporalAttention, self).__init__(**kwargs)

#     def build(self, input_shape):
#         """
#         input_shape: (batch_size, enc_seq_len, hidden_dim)
#         We'll learn a small dense layer to map from hidden_dim -> 1 (scores).
#         """
#         _, seq_len, hidden_dim = input_shape
#         self.score_dense = Dense(1)  # shape: (hidden_dim) -> (1)
#         super(TemporalAttention, self).build(input_shape)

#     def call(self, value):
#         """
#         value shape: (batch_size, enc_seq_len, hidden_dim)
#         Returns: context_vector shape (batch_size, hidden_dim)
#         """
#         # 1) Compute raw scores for each time step
#         #    shape => (batch_size, enc_seq_len, 1)
#         scores = self.score_dense(value)

#         # 2) Softmax over the 'enc_seq_len' dimension
#         attention_weights = tf.nn.softmax(scores, axis=1)  # (batch, enc_seq_len, 1)

#         # 3) Weighted sum of 'value'
#         context_vector = attention_weights * value  # (batch, enc_seq_len, hidden_dim)
#         context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, hidden_dim)

#         return context_vector