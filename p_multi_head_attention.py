#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention
from keras_core import ops
import tensorflow as tf

class PMultiHeadAttention(MultiHeadAttention):
  def __init__(
        self,
         **kwargs,
    ):
    super().__init__(
        **kwargs)


  def _compute_attention(
      self, query, key, value, attention_mask=None, training=None
  ):
      # B=1/0 - WoW - to działa :)
      # Note: Applying scalar multiply at the smaller end of einsum improves
      # XLA performance, but may introduce slight numeric differences in
      # the Transformer attention head.
      query = ops.multiply(
          query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
      )

      # Calculate attention scores using cosine similarity
      attention_scores = tf.keras.layers.Dot(axes=-1, normalize=True)([query, key])

      attention_scores = self._masked_softmax(
          attention_scores, attention_mask
      )

      # Apply dropout
      attention_scores_dropout = self._dropout_layer(
          attention_scores, training=training
      )

      # Compute the attention output
      attention_output = ops.einsum(
          self._combine_equation, attention_scores_dropout, value
      )
      return attention_output, attention_scores
