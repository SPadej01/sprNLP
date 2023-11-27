#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention
from keras_core import ops
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def smoothed_cosine_similarity(x, y, smoothing=1e-8):
    """
    Oblicza Smoothed Cosine Similarity między dwoma wektorami x i y.
    
    Args:
        x: Wektor 1.
        y: Wektor 2.
        smoothing: Wartość wygładzania (domyślnie 1e-8).
        
    Returns:
        Smoothed Cosine Similarity między x i y.
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
        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
        )

        # Obliczanie Smoothed Cosine Similarity między "query" i "key"
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

