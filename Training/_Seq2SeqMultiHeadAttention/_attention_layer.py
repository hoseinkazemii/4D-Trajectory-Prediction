import tensorflow as tf
from tensorflow.keras.layers import Layer, MultiHeadAttention, Reshape, LayerNormalization, Add, Dense, Dropout

class MultiHeadEncDecAttention(Layer):
    """
    Simple wrapper around Keras's MultiHeadAttention for seq2seq usage.
      - query shape: (batch_size, 1, hidden_dim) 
      - value shape: (batch_size, enc_seq_len, hidden_dim)
    Returns:
      context_vector shape: (batch_size, 1, hidden_dim)
    """
    def __init__(self, num_heads, key_dim, **kwargs):
        super(MultiHeadEncDecAttention, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        # self.layernorm = LayerNormalization()  # Add normalization
        self.add = Add() # Add residual connection
        # self.alpha = tf.Variable(1.0, trainable=True, name="attention_scaling_factor")  # Learnable scalar
        self.ffn = tf.keras.Sequential([
            Dense(128, activation='relu'),  # Intermediate layer
            Dropout(0.1),
            Dense(256)  # Project back to key_dim
        ])

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

        # context_vector = self.alpha * context_vector # Scale the context vector
        # Add residual connection (query + context_vector)
        context_vector = self.add([query, context_vector])

        # Apply feed-forward network
        context_vector = self.ffn(context_vector)

        return context_vector
