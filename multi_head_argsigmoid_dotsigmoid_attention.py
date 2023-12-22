#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention
from keras_core import ops


"""
Override the MultiHeadAttention class with use of sigmoid to calculate attention scores. 
In this implementation sigmoid function is applied to: keys, query and result of dot product.
"""
class MultiHeadArgSigmoidDotSigmoidAttention(MultiHeadAttention):
  def __init__(
        self,
         **kwargs,
    ):
    super().__init__(
        **kwargs)


    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        """Applies Sigmoid similarity attention between query, key, and dot product value tensors.

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

        # apply sigmoid on key and query
        key = ops.sigmoid(key)
        values = ops.sigmoid(values)
        # perform regular dot product
        attention_scores = ops.einsum(self._dot_product_equation, key, query)
        # Add sigmoid transformation to result of dot product
        attention_scores = ops.sigmoid(attention_scores) 

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        # Simple multiplication instead of softmax
        # if attention_mask is not None:
        #     attention_scores *= tf.expand_dims(attention_mask, axis=-1)

        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )

        attention_output = ops.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores