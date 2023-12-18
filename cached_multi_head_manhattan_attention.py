#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention
from keras_core import ops, backend
import tensorflow as tf
import numpy as np

"""
Override the MultiHeadAttention class using Euclidean distance
as similarity measure to  calculate attention scores.
"""

class CahedMultiHeadManhattanAttention(MultiHeadAttention):
    """MultiHeadAttention layer with cache support.

    This layer is suitable for use in autoregressive decoding. It can be used
    to cache decoder self-attention and cross-attention. The forward pass
    can happen in one of three modes:

    - No cache, same as regular multi-head attention.
    - Static cache (`cache_update_index` is None). In this case, the
        cached key/value projections will be used and the input values will
        be ignored.
    - Updated cache (`cache_update_index` is not None). In this case, new
        key/value projections are computed using the input, and spliced into
        the cache at the specified index.

    Note that caching is useful only during inference and should not be used
    during training.

    We use the notation `B`, `T`, `S` below, where `B` is the batch dimension,
    `T` is the target sequence length, and `S` in the source sequence length.
    Note that during generative decoding, `T` is usually 1 (you are
    generating a target sequence of length one to predict the next token).

    Call arguments:
        query: Query `Tensor` of shape `(B, T, dim)`.
        value: Value `Tensor` of shape `(B, S*, dim)`. if `cache` is None`, `S*`
            must equal `S` and match the shape of `attention_mask`. If cache` is
            not `None`, `S*` can be any length less than `S`, and the computed
            value will be spliced into `cache` at `cache_update_index`.
        key: Optional key `Tensor` of shape `(B, S*, dim)`. If `cache` is
            `None`, `S*` must equal `S` and match the shape of
            `attention_mask`. If `cache` is not `None`, `S*` can be any length
            less than `S`, and the computed value will be spliced into `cache`
            at `cache_update_index`.
        attention_mask: a boolean mask of shape `(B, T, S)`. `attention_mask`
            prevents attention to certain positions. The boolean mask specifies
            which query elements can attend to which key elements, 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        cache: a dense float Tensor. The key/value cache, of shape
            `[B, 2, S, num_heads, key_dims]`, where `S` must agree with the
            `attention_mask` shape. This argument is intended for use during
            generation to avoid recomputing intermediate state.
        cache_update_index: a int or int Tensor, the index at which to update
            `cache` (usually the index of the current token being processed
            when running generation). If `cache_update_index=None` while `cache`
            is set, the cache will not be updated.

    Returns:
        An `(attention_output, cache)` tuple. `attention_output` is the result
        of the computation, of shape `(B, T, dim)`, where `T` is for target
        sequence shapes and `dim` is the query input last dimension if
        `output_shape` is `None`. Otherwise, the multi-head outputs are
        projected to the shape specified by `output_shape`. `cache` is the
        updated cache.
    """

    def call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
    ):
        if (
            hasattr(self, "_build_from_signature")
            and not self._built_from_signature
        ):
            self._build_from_signature(query=query, value=value, key=key)

        if key is None:
            key = value

        query = self._query_dense(query)

        # If cache is not `None`, we will use the cache to compute the final key
        # and value tensors. If `cache_update_index` is not None, we will first
        # update the cache before use. To do this, we first call the
        # `_key_dense` and `_value_dense` layers, and copy the outputs into the
        # cache at the specified index. `cache = None` handles the training
        # case, where we don't use the cache at all.
        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                key_update = self._key_dense(key)
                value_update = self._value_dense(value)
                start = [0, cache_update_index, 0, 0]
                key = ops.slice_update(key_cache, start, key_update)
                value = ops.slice_update(value_cache, start, value_update)
                cache = ops.stack((key, value), axis=1)
        else:
            if cache_update_index is not None:
                raise ValueError(
                    "`cache_update_index` should not be set if `cache` is "
                    f"`None`. Received: cache={cache}, "
                    f"cache_update_index={cache_update_index}"
                )
            key = self._key_dense(key)
            value = self._value_dense(value)

        query = ops.multiply(
            query,
            1.0 / ops.sqrt(ops.cast(self._key_dim, query.dtype)),
        )
        # Zakładając, że query i key są tensorem o wymiarach (B, T, dim) i (B, S*, dim) odpowiednio
        query_expanded = ops.expand_dims(query, axis=-1)  # Dodajemy dodatkowy wymiar (B, T, dim, 1)
        key_expanded = ops.expand_dims(key, axis=-2)  # Dodajemy dodatkowy wymiar (B, S*, 1, dim)

        # Obliczamy odległość Manhattan między tensorem query i key
        manhattan_distance = ops.sum(ops.abs(query_expanded - key_expanded), axis=-1)  # Wynik: (B, T, S*)
        #manhattan_distance = backend.permute_dimensions(manhattan_distance, (0, 2, 1, 3))
        manhattan_distance = ops.transpose(manhattan_distance,(0, 2, 1, 3))  # transpozycja by uzyskać zgodność wymiarów z maską
        # attention_scores=manhattan_distance
        #keras.ops.transpose(x, axes=None)

        print(f'Shape of manhattan_distance:{manhattan_distance.shape}')
        attention_scores = manhattan_distance

        # Dot Product
        print(f'self._dot_product_equation: {self._dot_product_equation}')
        attention_scores = ops.einsum(self._dot_product_equation, key, query)
        print(f'Shape of dot product:{attention_scores.shape}')
        print(f'Shape of key:{key.shape}')
        print(f'Shape of query:{query.shape}')
        print(f'Shape of attention_mask:{attention_mask.shape}')
        


        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = self._dropout_layer(attention_scores)

        attention_output = ops.einsum(
            self._combine_equation, attention_scores, value
        )
        attention_output = self._output_dense(attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output