#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention
from keras_core import ops
#import tensorflow as tf
#import numpy as np

"""
Overriding the MultiHeadAttention class by applying cosine similarity
Cosine similarity is a measure of the similarity between two vectors that calculates the cosine of the angle between them.
The cosine similarity value ranges from -1 (completely different) to 1 (identical).

In the case of cosine similarity, the calculations are as follows:

1. Calculate the cosine similarity between query and key vectors.
2. Optionally, scale the cosine similarity results (e.g. by multiplying by a constant value).
3. Apply the softmax function to normalize the cosine similarity results.
4. Calculate self-attention values ​​by multiplying the normalized weights by the value vectors.

"""
class MultiHeadCosineAttention(MultiHeadAttention):
  def __init__(
        self,
         **kwargs,
    ):
    super().__init__(
        **kwargs)


    def _compute_attention(
            self, query, key, value, attention_mask=None, training=None
    ):
        """Applies Cosine Similarity attention with query, key, value tensors.

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
        # Remove scalling query because the query, key vectors are scalled using L2 norm

        # Queries and keys normalization
        # query = query / np.linalg.norm(query, axis=-1, keepdims=True)
        # key = key / np.linalg.norm(key, axis=-1, keepdims=True)

      # calculate L2 norm using ops functions 
        query_norm = ops.sqrt(ops.sum(ops.square(query), axis=-1, keepdims=True))
        key_norm = ops.sqrt(ops.sum(ops.square(key), axis=-1, keepdims=True))
        normalized_query = query / query_norm
        normalized_key = key / key_norm

        # Calculate cosine similairy between keys and queries
        attention_scores = ops.einsum(self._dot_product_equation, normalized_key, normalized_query)

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
