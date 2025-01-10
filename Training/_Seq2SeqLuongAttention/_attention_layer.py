import tensorflow as tf
from tensorflow.keras.layers import Layer

class Attention(Layer):
    """
    A simple Luong-style (dot-product) attention layer:
      - query shape: (batch_size, 1, hidden_dim)
      - value shape: (batch_size, enc_seq_len, hidden_dim)
    We learn a weight matrix W to transform `value` before computing dot-products with `query`.
    """
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        input_shape is a list/tuple:
          [ query_shape, value_shape ]
          query_shape = (batch_size, 1, hidden_dim)
          value_shape = (batch_size, enc_seq_len, hidden_dim)
        """
        _, value_shape = input_shape
        hidden_dim = value_shape[-1]

        # Luong-style: we transform the 'value' vectors with W before dot-product
        self.W = self.add_weight(
            name='W',
            shape=(hidden_dim, hidden_dim),
            initializer='glorot_uniform',
            trainable=True
        )

        super(Attention, self).build(input_shape)

    def call(self, inputs):
        """
        inputs = [query, value]
        query -> shape: (batch_size, 1, hidden_dim)
        value -> shape: (batch_size, enc_seq_len, hidden_dim)
        """
        query, value = inputs

        # 1) Transform the 'value' (encoder outputs) by W
        #    value: (batch_size, enc_seq_len, hidden_dim)
        #    W:     (hidden_dim, hidden_dim)
        # => value_transformed: (batch_size, enc_seq_len, hidden_dim)
        value_transformed = tf.tensordot(value, self.W, axes=[[2],[0]])
        
        # 2) Compute raw alignment scores via dot-product of value_transformed and query:
        #    value_transformed: (batch_size, enc_seq_len, hidden_dim)
        #    query:            (batch_size, 1, hidden_dim)
        #
        # => scores: (batch_size, enc_seq_len, 1)
        scores = tf.matmul(value_transformed, query, transpose_b=True)
        
        # 3) Softmax over the enc_seq_len dimension
        attention_weights = tf.nn.softmax(scores, axis=1)
        
        # 4. Compute context vector using batch matrix multiplication
        context_vector = tf.matmul(attention_weights, value, transpose_a=True)  # (batch_size, 1, hidden_dim)
        context_vector = tf.squeeze(context_vector, axis=1)  # (batch_size, hidden_dim)

        return context_vector

    def compute_output_shape(self, input_shape):
        # The output is (batch_size, hidden_dim)
        return (input_shape[0][0], input_shape[0][-1])




# from tensorflow.keras.layers import Layer
# import tensorflow as tf
# import math

# class Attention(Layer):
#     def __init__(self, **kwargs):
#         super(Attention, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # input_shape will be a list of shapes because we're dealing with multiple inputs
#         query_shape, value_shape = input_shape
#         self.W = self.add_weight(name="att_weight", shape=(value_shape[-1], value_shape[-1]), initializer="glorot_uniform", trainable=True)
#         self.b = self.add_weight(name="att_bias", shape=(value_shape[-1],), initializer="zeros", trainable=True)
#         super(Attention, self).build(input_shape)

#     def call(self, inputs):
#         query, value = inputs
#         # Compute the raw scores without applying tanh
#         score = tf.tensordot(query, self.W, axes=(2, 0)) / math.sqrt(self.W.shape[0]) + self.b
#         attention_weights = tf.nn.softmax(score, axis=1)
#         context_vector = attention_weights * value
#         context_vector = tf.reduce_sum(context_vector, axis=1)
#         return context_vector

#     def compute_output_shape(self, input_shape):
#         query_shape, value_shape = input_shape
#         return (query_shape[0], query_shape[-1])