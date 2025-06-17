import tensorflow as tf
from tensorflow.keras.layers import Layer, MultiHeadAttention

class LocalMultiHeadEncDecAttention(Layer):
    def __init__(self, window_size, key_dim, num_heads, dropout=0.1, **kwargs):
        super(LocalMultiHeadEncDecAttention, self).__init__(**kwargs)
        self.window_size = window_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.attention_weights = None
    
    def _build_local_attention_mask(self, query, value):
        batch_size = tf.shape(query)[0]
        T_q = tf.shape(query)[1]
        T_v = tf.shape(value)[1]
        
        i_idx = tf.range(T_q, dtype=tf.float32)
        j_idx = tf.range(T_v, dtype=tf.float32)
        
        i_idx_expanded = tf.reshape(i_idx, [-1, 1])
        j_idx_expanded = tf.reshape(j_idx, [1, -1])
        
        center_f = (i_idx_expanded * tf.cast(T_v - 1, tf.float32)) / tf.cast(T_q - 1, tf.float32)
        center_i = tf.round(center_f)
        
        dist = j_idx_expanded - center_i
        
        half_w = self.window_size // 2
        bool_mask = tf.abs(dist) >= half_w
        
        float_mask = tf.cast(bool_mask, tf.float32)
        
        additive_mask = (1.0 - float_mask) * -1e9
        
        additive_mask = tf.expand_dims(additive_mask, axis=0)
        additive_mask = tf.tile(additive_mask, [batch_size, 1, 1])
        
        return additive_mask
    
    def call(self, inputs, training=None):
        query, value = inputs
        
        attention_mask = self._build_local_attention_mask(query, value)
        
        attention_output, attention_scores = self.mha(
            query=query,
            value=value,
            key=value,
            attention_mask=attention_mask,
            return_attention_scores=True,
            training=training
        )
        
        self.attention_weights = attention_scores
        
        return attention_output
    
    def get_attention_weights(self):
        return self.attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "window_size": self.window_size,
            "key_dim": self.key_dim,
            "num_heads": self.num_heads,
        })
        return config
