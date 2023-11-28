#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention
from keras_core import ops
import tensorflow as tf
import numpy as np

"""
Nadpisanie klasy MultiHeadAttention poprzez dodanie funkcji tangensa hiperbolicznego
do obliczeń wyników podobieństwa ( attention scores ). 
"""
class MultiHeadTanhAttention(MultiHeadAttention):
  def __init__(
        self,
         **kwargs,
    ):
    super().__init__(
        **kwargs)


    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
        )

        # Do obliczeń funkji podobieństwa dodajemy przekształecenie przez tangens hiperboliczny
        attention_scores = tf.math.tanh(ops.einsum(self._dot_product_equation, key, query))

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