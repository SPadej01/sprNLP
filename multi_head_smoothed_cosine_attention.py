#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention
from keras_core import ops
import tensorflow as tf
import numpy as np

"""
Override the MultiHeadAttention class using Smoothed Cosine Similarity
as similarity measure to  calculate attention scores.
"""


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


        # query = ops.multiply(
        #     query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
        # )

        # Calculates Smoothed Cosine Similarity between "query" i "key"
        query_norm = ops.sqrt(ops.sum(ops.square(query), axis=-1, keepdims=True))
        key_norm = ops.sqrt(ops.sum(ops.square(key), axis=-1, keepdims=True))
        normalized_query = query / query_norm
        normalized_key = key / key_norm
        # calculate cosine similarity
        cosine_similarity = ops.einsum(self._dot_product_equation, normalized_key, normalized_query)
        # calculate smoothed cosine similarity
        epsilon=epsilon=1e-6
        attention_scores= (1 - cosine_similarity) / 2 + epsilon
        
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

