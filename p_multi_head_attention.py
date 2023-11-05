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
    a=0/0
    # Note: Applying scalar multiply at the smaller end of einsum improves
    # XLA performance, but may introduce slight numeric differences in
    # the Transformer attention head.
    query = ops.multiply(
        query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
    )


    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
#   attention_scores = ops.einsum(self._dot_product_equation, key, query)
   
    # Dot product changed to Cosine Similarity
    attention_scores = tf.keras.layers.Dot(axes=-1, normalize=True)([query, key])

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


def call(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):

        b=3/0
        
        if key is None:
            key = value

        attention_mask = self._compute_attention_mask(
            query,
            value,
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask, training
        )
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output
