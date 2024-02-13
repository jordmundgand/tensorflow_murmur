import tensorflow as tf

class RandomIndexing(tf.keras.layers.Layer):
  '''Preprocessing layer, returns single random index 
  among 1D sequence part,
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
  '''Preprocessing layer, replace single value in inputs[0] 
  to mask by index from inputs[1], useful for MLM'''
  def __init__(self, mask):
    super().__init__()
    def language_masking(x, masked_value=mask):
      return tf.vectorized_map(lambda z: tf.tensor_scatter_nd_update(z[0], z[1][tf.newaxis,:], [masked_value]), x)
    self.lambda0=tf.keras.layers.Lambda(language_masking)
  def call(self, inputs, training=None):
    masked=self.lambda0(inputs)
    return tf.keras.backend.in_train_phase(masked, inputs[0], training=training)

class IndexedSlice(tf.keras.layers.Layer):
  '''Preprocessing layer, returns single value from inputs[0]
  by index from inputs[1], useful for MLM'''
  def __init__(self):
    super().__init__()
    def indexed_slice(x):
      return tf.vectorized_map(lambda z: tf.gather(z[0], z[1]), x)
    self.lambda0=tf.keras.layers.Lambda(indexed_slice)
  def call(self, inputs):
    return self.lambda0(inputs)

class Splitter(tf.keras.layers.Layer):
    '''Preprocessing layer, splits input by sep to length,
    returns string tensor.'''
    def __init__(self,length=48,sep=' '):
        super().__init__()
        def multisplitter(x, length=length, sep=sep):
            out=tf.strings.split(x,sep=sep).to_tensor()
            pad=tf.cast(['']*length,dtype='string')
            out=tf.vectorized_map(lambda y: tf.concat((tf.squeeze(y),pad),axis=-1)[:length],out)
            return out
        self.length=length
        self.lambda0=tf.keras.layers.Lambda(multisplitter)
        self.reshape=tf.keras.layers.Reshape([self.length])
    def call(self, inputs):
        return self.reshape(self.lambda0(inputs))
    def compute_mask(self, inputs, mask=None):
        outputs=self.reshape(self.lambda0(inputs))
        return tf.not_equal(outputs, '')

class MultiText(tf.keras.layers.Layer):
    '''Preprocessing layer, convert string tensor into 2D count tensor 
    according to vocabulary.
    Parameters:
    vocabulary: list of tokens;
    length: int, layer length;
    max_tokens: max number of tokens;
    standardize: string or callable, like in TextVectorization;
    split: string or callable, like in TextVectorization;
    ngrams: like in TextVectorization;
    output_mode: string, like in TextVectorization;;
    pad_to_max_tokens: boolean, like in TextVectorization.'''
    def __init__(self, vocabulary, length, idf_weights=None, max_tokens=None, standardize='lower_and_strip_punctuation', split='whitespace',
                 ngrams=None, output_mode='count', pad_to_max_tokens=False):
        super().__init__()
        self.tv=tf.keras.layers.TextVectorization(max_tokens=max_tokens, standardize=standardize, 
        split=split, ngrams=ngrams, output_mode=output_mode, pad_to_max_tokens=pad_to_max_tokens, 
        vocabulary=vocabulary, idf_weights=idf_weights)
        self.concatenate=tf.keras.layers.Concatenate(axis=1)
        self.normalize=tf.keras.layers.Lambda(tf.nn.l2_normalize, arguments={'axis':-1})
        self.length=length

    def call(self, x):
        out=[self.tv(x[:,i][:,tf.newaxis])[:,tf.newaxis,:] for i in range(self.length)]
        out=self.concatenate(out)
        return self.normalize(out)
    def compute_mask(self, inputs, mask=None):
        return mask
