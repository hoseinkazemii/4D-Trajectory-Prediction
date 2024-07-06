from tensorflow.keras.layers import Layer
import tensorflow as tf
import math

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape will be a list of shapes because we're dealing with multiple inputs
        query_shape, value_shape = input_shape
        self.W = self.add_weight(name="att_weight", shape=(value_shape[-1], value_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(value_shape[-1],), initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        query, value = inputs
        # Compute the raw scores without applying tanh
        score = tf.tensordot(query, self.W, axes=(2, 0)) / math.sqrt(self.W.shape[0]) + self.b
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * value
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        query_shape, value_shape = input_shape
        return (query_shape[0], query_shape[-1])