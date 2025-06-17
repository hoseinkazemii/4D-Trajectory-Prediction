import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, Dropout
)


class MultiHeadEncDecAttention(Layer):
    def __init__(self, 
                 num_heads=4, 
                 key_dim=8,
                 ff_dim=128,
                 dropout=0.2, 
                 **kwargs):
        super(MultiHeadEncDecAttention, self).__init__(**kwargs)
        
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=key_dim,
            dropout=0.0
        )

        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(128)
        ])
        
        self.dropout = Dropout(dropout)
        self.attention_weights = None
        self.num_heads = num_heads
        self.key_dim = key_dim

    def call(self, inputs, training=None):
        query, value = inputs
        
        context_vector, attn_weights = self.mha(
            query=query,
            value=value,
            key=value, 
            return_attention_scores=True,
            training=training
        )
        self.attention_weights = attn_weights

        context_vector = self.ffn(self.dropout(context_vector, training=training))

        return context_vector

    def get_attention_weights(self):
        return self.attention_weights
