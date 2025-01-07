import tensorflow as tf
from tensorflow.keras.layers import Layer, MultiHeadAttention, Reshape, LayerNormalization

class MultiHeadEncDecAttention(Layer):
    """
    Simple wrapper around Keras's MultiHeadAttention for seq2seq usage.
      - query shape: (batch_size, 1, hidden_dim) 
      - value shape: (batch_size, enc_seq_len, hidden_dim)
    Returns:
      context_vector shape: (batch_size, 1, hidden_dim)
    """
    def __init__(self, num_heads=4, key_dim=16, **kwargs):
        super(MultiHeadEncDecAttention, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.layernorm = LayerNormalization()  # Add normalization


    def call(self, inputs, **kwargs):
        """
        inputs = [query, value]
          query => (batch_size, 1, hidden_dim)
          value => (batch_size, enc_seq_len, hidden_dim)
        """
        query, value = inputs

        # key = value (standard cross-attention pattern)
        # return_attention_scores=True => we can get the attn weights if needed
        context_vector, attn_weights = self.mha(
            query=query,
            value=value,
            key=value,
            return_attention_scores=True
        )
        # context_vector => (batch_size, 1, hidden_dim)

        self.attention_weights = attn_weights  # For visualization purposes

        return context_vector
