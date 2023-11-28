#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention
from keras_core import ops
import tensorflow as tf
import numpy as np


"""
Override the MultiHeadAttention class using Manhattan distance
as similarity measure to  calculate attention scores.
"""

class MultiHeadManhattanAttention(MultiHeadAttention):
  def __init__(
        self,
         **kwargs,
    ):



    super().__init__(
        **kwargs)


def _compute_attention(
    self, query, key, value, attention_mask=None, training=None
):
    """Applies Manhattan Distance Similarity attention with query, key, value tensors.

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
    # Compute the Manhattan distance between query and key tensors.
    attention_scores = tf.reduce_sum(tf.abs(query[:, :, :, :, None] - key[:, :, :, None, :, :]), axis=-1)

    attention_scores = self._masked_softmax(
        attention_scores, attention_mask
    )

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_scores_dropout = self._dropout_layer(
        attention_scores, training=training
    )

    # `context_layer` = [B, T, N, H]
    attention_output = ops.einsum(
        self._combine_equation, attention_scores_dropout, value
    )
    return attention_output, attention_scores
