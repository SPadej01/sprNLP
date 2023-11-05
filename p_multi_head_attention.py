#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention

class PMultiHeadAttention(MultiHeadAttention):
  def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
    super().__init__(
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        dropout=dropout,
        use_bias=use_bias,
        output_shape=output_shape,
        attention_axes=attention_axes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)

