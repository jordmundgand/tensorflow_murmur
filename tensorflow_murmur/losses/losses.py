import tensorflow as tf

def masked_loss(label, pred):
  '''Classical transformers masked SparseCategoricalCrossentropy 
  loss function.'''
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss

def masked_multi_loss(label, pred):
    '''Classical transformers masked CategoricalCrossentropy 
    loss function with sparse label input.'''
    label=tf.sparse.to_dense(label)
    mask = label == 0.
    mask = ~tf.math.reduce_all(mask,axis=-1)
    
    loss_object = tf.keras.losses.CategoricalCrossentropy(
      from_logits=False, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

class MaskedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    '''Classical transformers masked SparseCategoricalCrossentropy 
  loss.'''
    def __init__(self, from_logits=False, dtype=tf.float32, name="scc_m"):
        super().__init__(name=name)
        self.dtype=dtype 
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, reduction='none')
    def call(self, y_true, y_pred, sample_weight=None):
        mask=y_true!=0
        y_true=tf.cast(y_true,self.dtype)
        mask=tf.cast(mask,self.dtype)
        loss=self.loss_object(y_true, y_pred, sample_weight)
        loss*=mask
        return tf.math.divide_no_nan(tf.reduce_sum(loss),tf.reduce_sum(mask))
    def __call__(self,y_true, y_pred, sample_weight=None):
        return self.call(y_true, y_pred, sample_weight)

class WeightenedBinaryCrossentropy(tf.keras.losses.Loss):
    '''BinaryCrossentropy loss with auto-weightened classes for each batch. 
    Useful for imbalanced datasets. Only for sigmoid [None,1] output.
    Parameters:
    normalization_factor: 2 or else, if 2 the class is devided by sqrt(n_class+1)
    else by n_class;
    clip_by: float, clipping for log;
    dtype: str or dtype'''
    def __init__(self, normalization_factor=2, dtype=tf.float32, clip_by=1e-6, name="bce_w"):
        super().__init__(name=name)
        self.dtype=dtype 
        self.normalization_factor=normalization_factor
        self.clip_by=clip_by
    def call(self, y_true, y_pred, sample_weight=None):
        y_true=tf.cast(y_true,self.dtype)
        y_pred=tf.clip_by_value(y_pred, self.clip_by, 1.-self.clip_by)
        if not(sample_weight is None):
            term_0=tf.reduce_sum((1-y_true)*tf.math.log(1-y_pred)*sample_weight)
            term_1=tf.reduce_sum(y_true*tf.math.log(y_pred)*sample_weight)
        else:
            term_0=tf.reduce_sum((1-y_true)*tf.math.log(1-y_pred))
            term_1=tf.reduce_sum(y_true*tf.math.log(y_pred))
        if self.normalization_factor==2:
            term_0_m = tf.sqrt(tf.clip_by_value(tf.reduce_sum(1 - y_true)-1,0.,1e12))
            term_1_m = tf.sqrt(tf.clip_by_value(tf.reduce_sum(y_true)-1,0.,1e12))
        else:
            term_0_m = tf.reduce_sum(1 - y_true)
            term_1_m = tf.reduce_sum(y_true)
        return -(tf.math.divide_no_nan(term_0,term_0_m)+tf.math.divide_no_nan(term_1,term_1_m))
    def __call__(self,y_true, y_pred, sample_weight=None):
        return self.call(y_true, y_pred, sample_weight)
