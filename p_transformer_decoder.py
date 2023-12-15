# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cached_multi_head_cosine_attention import CachedCosineMultiHeadAttention
from multi_head_euclidean_attention import MultiHeadEuclideanAttention
from multi_head_smoothed_cosine_attention import MultiHeadSmoothedCosineAttention
from multi_head_manhattan_attention import MultiHeadManhattanAttention
from multi_head_tanh_attention import MultiHeadTanhAttention
from multi_head_sigmoid_attention import MultiHeadSigmoidAttention
from multi_head_sigmoidL1_attention import MultiHeadSigmoidL1Attention
from multi_head_sigmoid_noSM_attention import MultiHeadSigmoidNoSMAttention
from multi_head_sigmoid_sigmoid_noSM_attention import MultiHeadSigmoidSigmoidNoSMAttention


from keras_nlp.layers import TransformerDecoder
from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.backend import keras
from keras_nlp.src.backend import ops
from keras_nlp.src.layers.modeling.cached_multi_head_attention import (
    CachedMultiHeadAttention,
)

from keras_nlp.src.utils.keras_utils import clone_initializer


from keras_nlp.src.layers.modeling.transformer_layer_utils import (  # isort:skip
    compute_causal_mask,
    merge_padding_and_attention_mask,
)


attentions = {
            "DOT":CachedMultiHeadAttention,
            "Cosine":CachedCosineMultiHeadAttention,
            "Euclidean":MultiHeadEuclideanAttention,
            "SmoothedCosine":MultiHeadSmoothedCosineAttention,
            "Manhattan":MultiHeadManhattanAttention,
            "Tanh":MultiHeadTanhAttention,
            "Sigmoid":MultiHeadSigmoidAttention,
            "SigmoidL1":MultiHeadSigmoidL1Attention,
            "SigmoidNoSM":MultiHeadSigmoidNoSMAttention,
            "SigmoidSigmoidNoSM":MultiHeadSigmoidSigmoidNoSMAttention,
            

            }


class PTransformerDecoder(TransformerDecoder):
    """Transformer decoder.

    This class follows the architecture of the transformer decoder layer in the
    paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users
    can instantiate multiple instances of this class to stack up a decoder.

    By default, this layer will apply a causal mask to the decoder attention layer.
    This layer will correctly compute an attention mask from an implicit
    Keras padding mask (for example, by passing `mask_zero=True` to a
    `keras.layers.Embedding` layer). See the Masking and Padding
    [guide](https://keras.io/guides/understanding_masking_and_padding/)
    for more details.

    This layer can be called with either one or two inputs. The number of inputs
    must be consistent across all calls. The options are as follows:
        `layer(decoder_sequence)`: no cross-attention will be built into the
            decoder block. This is useful when building a "decoder-only"
            transformer such as GPT-2.
        `layer(decoder_sequence, encoder_sequence)`: cross-attention will be
            built into the decoder block. This is useful when building an
            "encoder-decoder" transformer, such as the original transformer
            model described in Attention is All You Need.

    Args:
        intermediate_dim: int, the hidden size of feedforward network.
        num_heads: int, the number of heads in MultiHeadAttention.
        dropout: float. the dropout value, shared by
            MultiHeadAttention and feedforward network. Defaults to `0.`.
        activation: string or `keras.activations`. the
            activation function of feedforward network.
            Defaults to `"relu"`.
        layer_norm_epsilon: float. The eps value in layer
            normalization components. Defaults to `1e-5`.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense and multiheaded
            attention layers. Defaults to `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense and multiheaded
            attention layers. Defaults to `"zeros"`.
        normalize_first: bool. If True, the inputs to the
            attention layer(s) and the intermediate dense layer are normalized
            (similar to GPT-2). If set to False, outputs of attention layer and
            intermediate dense layer are normalized (similar to BERT).
            Defaults to `False`.
        name: string. The name of the layer. Defaults to `None`.
        **kwargs: other keyword arguments.

    Examples:
    ```python
    # Create a single transformer decoder layer.
    decoder = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=64, num_heads=8)

    # Create a simple model containing the decoder.
    decoder_input = keras.Input(shape=(10, 64))
    encoder_input = keras.Input(shape=(10, 64))
    output = decoder(decoder_input, encoder_input)
    model = keras.Model(
        inputs=(decoder_input, encoder_input),
        outputs=output,
    )

    # Call decoder on the inputs.
    decoder_input_data = np.random.uniform(size=(2, 10, 64))
    encoder_input_data = np.random.uniform(size=(2, 10, 64))
    decoder_output = model((decoder_input_data, encoder_input_data))
    ```

    References:
     - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

    """

    def __init__(self,
      attention_type,
        **kwargs,    ):
        self.attention_type=attention_type
        super().__init__(
          **kwargs)

    def build(
        self,
        decoder_sequence_shape,
        encoder_sequence_shape=None,
    ):
        self._decoder_sequence_shape = decoder_sequence_shape
        self._encoder_sequence_shape = encoder_sequence_shape
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = decoder_sequence_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        head_dim = int(hidden_dim // self.num_heads)
        if head_dim == 0:
            raise ValueError(
                "Attention `head_dim` computed cannot be zero. "
                f"The `hidden_dim` value of {hidden_dim} has to be equal to "
                f"or greater than `num_heads` value of {self.num_heads}."
            )
        # pobierz klase atencji bazując na parametrze
        if self.attention_type in attentions.keys():
          attention_class = attentions[self.attention_type]
          print(f"Wyznaczono klasę atencji dla: {self.attention_type}: {attention_class}")
        else:
          raise ValueError(f"Nie wyznaczono klasy atencji dla parametru: {self.attention_type}")

        # Wywołaj wskazaną w parametrze klasę atencji
        self._self_attention_layer = attention_class ( 
            num_heads=self.num_heads,
            key_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name=f"attention_{self.attention_type}",
        )


        # Self attention layers.
        # self._self_attention_layer = CachedMultiHeadAttention(
        #     num_heads=self.num_heads,
        #     key_dim=head_dim,
        #     dropout=self.dropout,
        #     kernel_initializer=clone_initializer(self.kernel_initializer),
        #     bias_initializer=clone_initializer(self.bias_initializer),
        #     dtype=self.dtype_policy,
        #     name="self_attention",
        # )

        
        if hasattr(self._self_attention_layer, "_build_from_signature"):
            self._self_attention_layer._build_from_signature(
                query=decoder_sequence_shape,
                value=decoder_sequence_shape,
            )
        else:
            self._self_attention_layer.build(
                query_shape=decoder_sequence_shape,
                value_shape=decoder_sequence_shape,
            )
        self._self_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self._self_attention_layer_norm.build(decoder_sequence_shape)
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        # Cross attention layers are optional.
        self._cross_attention_layer = None
        if encoder_sequence_shape:
            self._cross_attention_layer = attention_class ( 
              num_heads=self.num_heads,
                  key_dim=head_dim,
                  value_dim=head_dim,
                  dropout=self.dropout,
                  kernel_initializer=clone_initializer(self.kernel_initializer),
                  bias_initializer=clone_initializer(self.bias_initializer),
                  dtype=self.dtype_policy,
              name=f"attention_{self.attention_type}",
          )
            # self._cross_attention_layer = CachedMultiHeadAttention(
            #     num_heads=self.num_heads,
            #     key_dim=head_dim,
            #     value_dim=head_dim,
            #     dropout=self.dropout,
            #     kernel_initializer=clone_initializer(self.kernel_initializer),
            #     bias_initializer=clone_initializer(self.bias_initializer),
            #     dtype=self.dtype_policy,
            #     name="cross_attention",
            # )
            if hasattr(self._cross_attention_layer, "_build_from_signature"):
                self._cross_attention_layer._build_from_signature(
                    query=encoder_sequence_shape,
                    value=encoder_sequence_shape,
                )
            else:
                self._cross_attention_layer.build(
                    query_shape=encoder_sequence_shape,
                    value_shape=encoder_sequence_shape,
                )
            self._cross_attention_layer_norm = keras.layers.LayerNormalization(
                epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="cross_attention_layer_norm",
            )
            self._cross_attention_layer_norm.build(encoder_sequence_shape)
            self._cross_attention_dropout = keras.layers.Dropout(
                rate=self.dropout,
                dtype=self.dtype_policy,
                name="cross_attention_dropout",
            )

        # Feedforward layers.
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(decoder_sequence_shape)
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        intermediate_shape = list(decoder_sequence_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._feedforward_output_dense.build(tuple(intermediate_shape))
        self._feedforward_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self._feedforward_layer_norm.build(decoder_sequence_shape)
        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )
        # Create layers based on input shape.
        self.built = True


