import tensorflow as tf
import numpy as np

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
  
  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

class IdfEmbedding(tf.keras.layers.Layer):
    '''Like usal Embedding, but applying idf weights to final representation. 
    Parameters:
    input_dim: int, vocabulary dimension;
    output_dim: int, embedding dimension;
    idf: iterrable, vector of idf weights.'''
    def __init__(
        self,
        input_dim,
        output_dim,
        idf,
        mask_zero=True,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        activity_regularizer=None,
        embeddings_constraint=None,
        **kwargs
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.idf = tf.constant(idf, dtype=tf.dtypes.float32)[:,tf.newaxis]
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.embeddings_constraint = tf.keras.constraints.get(embeddings_constraint)
        self.mask_zero=mask_zero
        super(IdfEmbedding, self).__init__(**kwargs)
    
    def build(self, input_shape=None):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name="idf_embeddings",
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint)
        self.built = True

    def call(self, inputs):
        dtype = tf.keras.backend.dtype(inputs)
        if dtype != "int32" and dtype != "int64":
            inputs = tf.cast(inputs, "int32")
        
        out = tf.nn.embedding_lookup(self.embeddings, inputs)
        idfs = tf.nn.embedding_lookup(self.idf, inputs)
        out = out*idfs
        if (
            self._dtype_policy.compute_dtype
            != self._dtype_policy.variable_dtype
        ):
            # Instead of casting the variable as in most layers, cast the
            # output, as this is mathematically equivalent but is faster.
            out = tf.cast(out, self._dtype_policy.compute_dtype)
        return out
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, 0)

class PositionalEmbedding(tf.keras.layers.Layer):
  '''Classical transformers positional embedding layer. 
  Parameters:
  vocab_size: int, vocabulary dimension;
  d_model: int, embedding dimension;
  length: int, length of the encoding sequence'''
  def __init__(self, vocab_size, d_model, length=2048):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=length, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

class SpatialEmbedding(tf.keras.layers.Layer):
  '''Transformers positional embedding layer, 
  encoding continuous values into embeddings. 
  Input: 2D float or int tensor(N x M), M in [1: small int],
  output: 2D float or int tensor (N x d_model).
  Parameters:
  d_model: int, embedding dimension;
  activation: string or callable, inner dense activation;
  mask_value: float, value to mask.'''
  
  def __init__(self, d_model, activation='tanh', mask_value=0.0):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Dense(d_model, activation=activation)
    self.pos_encoding = positional_encoding(length=128, depth=d_model)
    self.mask_value = mask_value

  def call(self, x):
    boolean_mask = tf.reduce_any(
            tf.not_equal(x, self.mask_value), axis=-1, keepdims=True)
    length = tf.shape(x)[1]
    x = self.embedding(x)

    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    x._keras_mask = tf.squeeze(boolean_mask, axis=-1)
    return x

  def compute_mask(self, inputs, mask=None):
        return tf.reduce_any(tf.not_equal(inputs, self.mask_value), axis=-1)
  
  def compute_output_shape(self, input_shape):
        return (input_shape[0], self.d_model)
  
