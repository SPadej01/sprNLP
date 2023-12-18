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

        query_expanded = ops.expand_dims(query, axis=2)  
        key_expanded = ops.expand_dims(key, axis=1) 

        # Obliczamy kwadrat różnicy między tensorem query i key
        squared_diff = ops.square(query_expanded - key_expanded)  # Wynik: (B, T, S*, dim)

        # Sumujemy kwadraty różnic wzdłuż ostatniego wymiaru (dim)
        sum_squared_diff = ops.sum(squared_diff, axis=-1)  # Wynik: (B, T, S*)

        # Obliczamy pierwiastek kwadratowy z sumy kwadratów różnic
        euclidean_distance = ops.sqrt(sum_squared_diff)  # Wynik: (B, T, S*)

        # Transponujemy wynik, aby uzyskać wymiary (B, S*, T, T)
        euclidean_distance = ops.transpose(euclidean_distance, (0, 3,1,2))  # Wynik: (B, 8, 40, 40)
        
                
        # dot_product = ops.einsum(self._dot_product_equation, key, query)

        attention_scores= euclidean_distance


        # print(f'Shape of key:{key.shape}')
        # print(f'Shape of query:{query.shape}')
        # print(f'Shape of key_expanded:{key_expanded.shape}')
        # print(f'Shape of query_expanded:{query_expanded.shape}')
        
        # print(f'Shape of query:{query.shape}')
        # print(f'Shape of euclidean_distance:{euclidean_distance.shape}')
        # print(f'self._dot_product_equation:{self._dot_product_equation}')
        # print(f'Shape of dot_product:{dot_product.shape}')


        # Normalizacja wyników uwagi
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = self._dropout_layer(attention_scores)

        attention_output = ops.einsum(
            self._combine_equation, attention_scores, value
        )
        attention_output = self._output_dense(attention_output)
        return attention_output

