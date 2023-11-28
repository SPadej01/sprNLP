#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention
from keras_core import ops
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

"""
Override the MultiHeadAttention class using Smoothed Cosine Similarity
as similarity measure to  calculate attention scores.
"""


def smoothed_cosine_similarity(x, y, smoothing=1e-8):
    """
  Calculates Smoothed Cosine Similarity between two vectors x and y.
    
    Args:
        x: Vector 1.
        y: Vector 2.
        smoothing: Smoothing value (default 1e-8).
        
    Returns:
        Smoothed Cosine Similarity between x and y.
    """
    return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)) + smoothing



class MultiHeadSmoothedCosineAttention(MultiHeadAttention):
  def __init__(
        self,
         **kwargs,
    ):



    super().__init__(
        **kwargs)



    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        """Applies Smoothed Cosine Similarity attention with query, key, value tensors.

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


        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
        )

        # Calculates Smoothed Cosine Similarity between "query" i "key"
        attention_scores = np.zeros(
            (query.shape[0], query.shape[1], key.shape[1], key.shape[2])
        )
        for b in range(query.shape[0]):
            for t in range(query.shape[1]):
                for s in range(key.shape[1]):
                    for h in range(key.shape[2]):
                        attention_scores[b, t, s, h] = smoothed_cosine_similarity(
                            query[b, t, h], key[b, s, h]
                        )

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )

        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )

        attention_output = ops.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores

