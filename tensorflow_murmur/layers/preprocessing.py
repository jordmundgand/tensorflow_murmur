import tensorflow as tf

class RandomIndexing(tf.keras.layers.Layer):
  '''Preprocessing layer, returns single random index among 1D sequence part,
     which is not == mask_value, useful for MLM'''
  def __init__(self, mask_value=0):
    super().__init__()
    def random_masked_indexing(x, mask_value=mask_value):
        x=x!=mask_value
        x=tf.cast(x,dtype=tf.int32)
        x=tf.reduce_sum(x,axis=-1,keepdims=True)
        return tf.vectorized_map(lambda y: tf.random.uniform([1], minval=0, maxval=y[0], dtype=tf.int32), x)
      self.lambda0=tf.keras.layers.Lambda(random_masked_indexing)
  def call(self, inputs):     
    return self.lambda0(inputs)

class LanguageMasking(tf.keras.layers.Layer):
  '''Preprocessing layer, replace single value in inputs[0] to mask by index from inputs[1], useful for MLM'''
  def __init__(self, mask):
    super().__init__()
    def language_masking(x, masked_value=mask):
      return tf.vectorized_map(lambda z: tf.tensor_scatter_nd_update(z[0], z[1][tf.newaxis,:], [masked_value]), x)
    self.lambda0=tf.keras.layers.Lambda(language_masking)
  def call(self, inputs, training=None):
    masked=self.lambda0(inputs)
    return tf.keras.backend.in_train_phase(masked, inputs[0], training=training)

class IndexedSlice(tf.keras.layers.Layer):
  '''Preprocessing layer, returns single value from inputs[0] by index from inputs[1], useful for MLM'''
  def __init__(self):
    super().__init__()
    def indexed_slice(x):
      return tf.vectorized_map(lambda z: tf.gather(z[0], z[1]), x)
    self.lambda0=tf.keras.layers.Lambda(indexed_slice)
  def call(self, inputs):
    return self.lambda0(inputs)
