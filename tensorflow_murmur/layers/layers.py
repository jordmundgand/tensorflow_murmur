import tensorflow as tf
import numpy as np

class LSTMTransformerLayer(tf.keras.layers.Layer):
  '''Transformers encoder-like layer, similar to EncoderLayer, 
  but based on residual bidirectional LSTM
  instead of MultiHeadAttention and FeedForward. 
  Doesn`t require positional encoding.
  Parameters:
  height: int, embedding dimension;
  dropout: 0<float<1 LSTM dropout between layers'''
  def __init__(self, height, dropout=0):
    super().__init__()
    self.LSTMb=tf.keras.layers.LSTM(units=height, return_sequences=True, 
                                    go_backwards=True,dropout=dropout)
    self.LSTMf=tf.keras.layers.LSTM(units=height, return_sequences=True,
                                   dropout=dropout)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
  def call(self, x):
    lstmb=self.LSTMb(x)
    lstmf=self.LSTMf(x)
    x = self.add([x, lstmb, lstmf])
    x = self.layernorm(x)
    return x

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

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)
   
    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, activation='gelu', dropout_rate=0.0):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation=activation),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x

class EncoderLayer(tf.keras.layers.Layer):
  '''Classical transformers encoder layer,  based on MultiHeadAttention 
  and FeedForward. 
  Parameters:
  d_model: int, embedding dimension;
  num_heads: int, number of attention heads;
  dff: int, FeedForward inner dimension;
  activation: str or function, inner FeedForward layer activation;
  attention_dropout: 0<float<1 inner MultiHeadAttention layer dropout
  ffn_dropout: 0<float<1 inner inner FeedForward layer dropout'''
  def __init__(self,*, d_model, num_heads, dff, activation='gelu', attention_dropout=0.0, ffn_dropout=0.0):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=attention_dropout)

    self.ffn = FeedForward(d_model, dff, activation=activation, dropout_rate=ffn_dropout)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class DecoderLayer(tf.keras.layers.Layer):
  '''Classical transformers decoder layer,  based on MultiHeadAttention 
  and FeedForward. 
  Parameters:
  d_model: int, embedding dimension;
  num_heads: int, number of attention heads;
  dff: int, FeedForward inner dimension;
  activation: str or function, inner FeedForward layer activation;
  attention_dropout: 0<float<1 inner MultiHeadAttention layer dropout
  ffn_dropout: 0<float<1 inner inner FeedForward layer dropout'''
  def __init__(self,
               *,
               d_model, num_heads, dff, activation='gelu', attention_dropout=0.0, ffn_dropout=0.0):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=attention_dropout)
    
    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=attention_dropout)

    self.ffn = FeedForward(d_model, dff, activation=activation, dropout_rate=ffn_dropout)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x

def random_masked_indexing(x):
    x=tf.reduce_sum(x,axis=-1,keepdims=True)
    return tf.vectorized_map(lambda y: tf.random.uniform([1], minval=0, maxval=y[0], dtype=tf.int32), x)
  
def language_masking(x, masked_value=2):
    return tf.vectorized_map(lambda z: tf.tensor_scatter_nd_update(z[0], z[1][tf.newaxis,:], [masked_value]), x)

class Weightened(tf.keras.layers.Layer):
    '''Adaptive weights layer (trainable TF-IDF) 
    Parameters:
    units: tuple, shape of weights tensor (without batch dimension), must be equal to previous layer;
    initializer: initializer, default - ones;
    dtype: str or dtype;
    trainable: bool;'''
    def __init__(self, units=(1,), initializer=tf.keras.initializers.Ones(), 
                 dtype='float32', trainable=True):
        super(Weightened, self).__init__()
        self.units=units
        self.initializer=initializer
        self.dtype_=dtype
        self.trainable=trainable
        self.w = tf.Variable(name="weights",
        initial_value=self.initializer(shape=(1,)+ self.units, dtype=dtype),
        trainable=trainable)

    def build(self, input_shape):
        self.w = tf.Variable(name="weights",
        initial_value=self.initializer(shape=(1,)+ self.units, dtype=self.dtype_),
        trainable=self.trainable)

    def call(self, inputs):
        inputs=tf.cast(inputs,dtype=self.dtype_)
        return inputs*self.w
