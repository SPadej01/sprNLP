from keras_nlp.layers import TransformerEncoder
from multi_head_cosine_attention import MultiHeadCosineAttention
from multi_head_euclidean_attention import MultiHeadEuclideanAttention
from multi_head_smoothed_cosine_attention import MultiHeadSmoothedCosineAttention
from multi_head_manhattan_attention import MultiHeadManhattanAttention
from multi_head_tanh_attention import MultiHeadTanhAttention
from multi_head_dotsigmoid_attention import MultiHeadDotSigmoidAttention
from multi_head_argigmoid_dotsigmoid_attention import MultiHeadArgSigmoidDotSigmoidAttention



from keras_nlp.src.utils.keras_utils import clone_initializer
from keras_nlp.src.backend import keras




attentions = {"DOT":keras.layers.MultiHeadAttention,
            "Cosine":MultiHeadCosineAttention,
            "Euclidean":MultiHeadEuclideanAttention,
            "SmoothedCosine":MultiHeadSmoothedCosineAttention,
            "Manhattan":MultiHeadManhattanAttention,
            "Tanh":MultiHeadTanhAttention,
            "DotSigmoid":MultiHeadDotSigmoidAttention,
            "ArgSigmoidDotSigmoid":MultiHeadArgSigmoidDotSigmoidAttention,
            }




class PTransformerEncoder(TransformerEncoder):
      def __init__(self,
      attention_type,
        **kwargs,    ):
        self.attention_type=attention_type
        super().__init__(
          **kwargs)

      def build(self, inputs_shape):
  
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = inputs_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        key_dim = int(hidden_dim // self.num_heads)
        if key_dim == 0:
            raise ValueError(
                "Attention `key_dim` computed cannot be zero. "
                f"The `hidden_dim` value of {hidden_dim} has to be equal to "
                f"or greater than `num_heads` value of {self.num_heads}."
            )
        self._name = f"{self.__class__.__name__}_{self.attention_type}"

        # pobierz klase atencji bazując na parametrze
        if self.attention_type in attentions.keys():
          attention_class = attentions[self.attention_type]
        else:
          raise ValueError(f"Nie wyznaczono klasy atencji dla parametru: {self.attention_type}")

        # Wywołaj wskazaną w parametrze klasę atencji
        self._self_attention_layer = attention_class ( 
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name=f"attention_{self.attention_type}",
        )
        if hasattr(self._self_attention_layer, "_build_from_signature"):
            self._self_attention_layer._build_from_signature(
                query=inputs_shape,
                value=inputs_shape,
            )
        else:
            self._self_attention_layer.build(
                query_shape=inputs_shape,
                value_shape=inputs_shape,
            )
        self._self_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self._self_attention_layer_norm.build(inputs_shape)
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        # Feedforward layers.
        self._feedforward_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self._feedforward_layer_norm.build(inputs_shape)
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(inputs_shape)
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        intermediate_shape = list(inputs_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._feedforward_output_dense.build(tuple(intermediate_shape))
        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )
        self.built = True
