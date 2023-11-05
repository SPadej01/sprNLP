from keras_nlp.layers import TransformerEncoder

class PTransformerEncoder(TransformerEncoder):
      def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0,
        activation="relu",
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        normalize_first=False,
        **kwargs,
    ):
        super().__init__(
          intermediate_dim=intermediate_dim,
          num_heads=num_heads,
          dropout=dropout,
          activation=activation,
          layer_norm_epsilon=layer_norm_epsilon,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          normalize_first=normalize_first,
          **kwargs)