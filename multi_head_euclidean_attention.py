#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention
from keras_core import ops
import tensorflow as tf
import numpy as np

"""
Override the MultiHeadAttention class using Euclidean distance
as similarity measure to  calculate attention scores.
"""

class MultiHeadEuclideanAttention(MultiHeadAttention):

  def __init__(
        self,
         **kwargs,
    ):
    super().__init__(
        **kwargs)

    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):

        """Applies Euclidean Distance Similarity attention with query, key, value tensors.

            Args:
                query: Projected query tensor of shape `(B, T, N, key_dim)`.
                key: Projected key tensor of shape `(B, S, N, key_dim)`.
                value: Projected value tensor of shape `(B, S, N, value_dim)`.
                attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                    attention to certain positions. It is generally not needed if
                    the `query` and `value` (and/or `key`) are masked.
                training: Python boolean indicating whether the layer should behave
                    in training mode (adding dropout) or in inference mode (no dropout).

            Returns:
              attention_output: Multi-headed outputs of attention computation.
              attention_scores: Multi-headed attention weights.
        """

        # Calculating the Euclidean distance between keys and queries
        euclidean_distance = tf.sqrt(tf.reduce_sum(tf.square(query[:, None, :, :] - key[:, :, None, :]), axis=-1))

        # Apply the attention mask if provided
        if attention_mask is not None:
            euclidean_distance += (1 - attention_mask) * 1e9

        # Apply the softmax function to obtain attention probabilities
        attention_probs = tf.nn.softmax(-euclidean_distance, axis=-1)

        # Apply dropout if specified
        if self._dropout > 0.0:
            attention_probs = tf.nn.dropout(attention_probs, rate=self._dropout)

        # Determination of the weighted sum of vectors of values ​​and attention probabilities
        attention_output = tf.matmul(attention_probs, value)

        return attention_output, attention_probs

