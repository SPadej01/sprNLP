#from keras.layers import MultiHeadAttention
#Trzeba być zgodnym z Keras 3.0....
from keras_core.src.layers import MultiHeadAttention
import tensorflow as tf

class MultiHeadEuclideanAttention(MultiHeadAttention):
    def __init__(self,**kwargs,):
        super().__init__(**kwargs)
    """
      Obliczamy odległość euklidesową między wektorami `query` i `key`. 
      Następnie, stosujemy maskę uwagi, jeśli została dostarczona, i funkcję softmax, aby uzyskać prawdopodobieństwa uwagi. 
      Ostatecznie, obliczamy wynikową macierz atencji jako ważoną sumę wektorów wartości, 
      gdzie wagi są określone przez prawdopodobieństwa uwagi.
     """
    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        # Compute the Euclidean distance between the query and key vectors
        euclidean_distance = tf.sqrt(tf.reduce_sum(tf.square(query[:, None, :, :] - key[:, :, None, :]), axis=-1))
    
        # Apply the attention mask if provided
        if attention_mask is not None:
            euclidean_distance += (1 - attention_mask) * 1e9
    
        # Apply the softmax function to obtain attention probabilities
        attention_probs = tf.nn.softmax(-euclidean_distance, axis=-1)
    
        # Apply dropout if specified
        if self._dropout > 0.0:
            attention_probs = tf.nn.dropout(attention_probs, rate=self._dropout)
    
        # Compute the weighted sum of the value vectors using the attention probabilities
        attention_output = tf.matmul(attention_probs, value)
    
        return attention_output, attention_probs