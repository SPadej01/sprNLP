#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention
from keras_core import ops
import tensorflow as tf
import numpy as np

"""
Override the MultiHeadAttention class with use of sigmoid and L1 norm to calculate attention scores. 
"""

class MultiHeadSigmoidL1Attention(MultiHeadAttention):
  def __init__(
        self,
         **kwargs,
    ):
    super().__init__(
        **kwargs)

  def _compute_attention(
    self, query, key, value, attention_mask=None, training=None
):
    """Applies Sigmoid with L1 norm similarity attention between query, key, value tensors.

    This function defines the computation inside `call` with projected
    multi-head Q, K, V inputs. Users can override this function for
    customized attention implementation.

    Args:
        query: Projected query tensor of shape `(B, T, N, key_dim)`.
        key: Projected key tensor of shape `(B, S, N, key_dim)`.
        value: Projected value tensor of shape `(B, S, N, value_dim)`.
        attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions. It is generally not needed if
            the `query` and `value` (and/or `key`) are masked.
        training: Python boolean indicating whether the layer should behave
            in training mode (adding dropout) or in inference mode (doing
            nothing).

    Returns:
      attention_output: Multi-headed outputs of attention computation.
      attention_scores: Multi-headed attention weights.
    """
    query = ops.multiply(
        query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
    )

    attention_scores = ops.einsum(self._dot_product_equation, key, query)

    attention_scores = tf.math.sigmoid(attention_scores)  # Use sigmoid

    # Normalize attention scores using L1 norm
    attention_scores /= tf.reduce_sum(attention_scores, axis=-1, keepdims=True)

    if attention_mask is not None:
        attention_scores *= ops.cast(attention_mask, attention_scores.dtype)

    attention_scores_dropout = self._dropout_layer(
        attention_scores, training=training
    )

    attention_output = ops.einsum(
        self._combine_equation, attention_scores_dropout, value
    )
    return attention_output, attention_scores