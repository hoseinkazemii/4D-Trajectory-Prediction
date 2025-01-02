import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class TemporalAttention(Layer):
    """
    Learns a per-timestep weight distribution (attention) from the encoder outputs.
    No explicit 'query' is used. This is a simpler 'temporal-only' attention.
    """
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        input_shape: (batch_size, enc_seq_len, hidden_dim)
        We'll learn a small dense layer to map from hidden_dim -> 1 (scores).
        """
        _, seq_len, hidden_dim = input_shape
        self.score_dense = Dense(1)  # shape: (hidden_dim) -> (1)
        super(TemporalAttention, self).build(input_shape)

    def call(self, value):
        """
        value shape: (batch_size, enc_seq_len, hidden_dim)
        Returns: context_vector shape (batch_size, hidden_dim)
        """
        # 1) Compute raw scores for each time step
        #    shape => (batch_size, enc_seq_len, 1)
        scores = self.score_dense(value)

        # 2) Softmax over the 'enc_seq_len' dimension
        attention_weights = tf.nn.softmax(scores, axis=1)  # (batch, enc_seq_len, 1)

        # 3) Weighted sum of 'value'
        context_vector = attention_weights * value  # (batch, enc_seq_len, hidden_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, hidden_dim)

        return context_vector