#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention
from keras_core import ops


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

        query_expanded = ops.expand_dims(query, axis=2)  
        key_expanded = ops.expand_dims(key, axis=1) 

        # square of the difference between the query and key tensors
        squared_diff = ops.square(query_expanded - key_expanded)  # Result: (B, T, S*, dim)

        #  sum the squares of differences along the last dimension (dim)
        sum_squared_diff = ops.sum(squared_diff, axis=-1)  # Result: (B, T, S*)

        # the square root of the sum of the squares of the differences
        euclidean_distance = ops.sqrt(sum_squared_diff)  # Result: (B, T, S*)

        # Transpose results to acquire dimension (B, S*, T, T)
        euclidean_distance = ops.transpose(euclidean_distance, (0, 3,1,2))  # Result: (B, 8, 40, 40)
        
                
        # dot_product = ops.einsum(self._dot_product_equation, key, query)
        attention_scores= euclidean_distance

        # Normalize and mask results
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = self._dropout_layer(attention_scores)

        attention_output = ops.einsum(
            self._combine_equation, attention_scores, value
        )
        attention_output = self._output_dense(attention_output)
        return attention_output

