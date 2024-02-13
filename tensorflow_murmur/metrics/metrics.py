import tensorflow as tf

def masked_accuracy(label, pred):
  '''Classical transformers masked SparseCategoricalAccuracy 
  metric function.'''
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

def masked_multi_cosine_similarity(label, pred):
    '''Masked CosineSimilarity metric function
    with sparse tensor label input.'''
    label=tf.sparse.to_dense(label)
    mask = label == 0.
    mask = ~tf.math.reduce_all(mask,axis=-1)
    
    label = tf.linalg.l2_normalize(label, axis=-1)
    pred = tf.linalg.l2_normalize(pred, axis=-1)
    metric = tf.reduce_sum(label*pred, axis=-1)

    mask = tf.cast(mask, dtype=metric.dtype)
    metric *= mask

    metric = tf.reduce_sum(metric)/tf.reduce_sum(mask)
    return metric
  
def sparse_target_accuracy(label, pred):
  '''Accuracy function for full sequenses identity,
  sparse labels.'''
  pred = tf.argmax(pred, axis=-1)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  match = tf.cast(match, dtype=tf.float32)
  match = tf.math.reduce_min(match, axis=-1)
  return tf.reduce_mean(match)

def batch_average(label, pred):
  '''Special `accuracy` function, useful for MLM.'''
  return tf.reduce_mean(pred)
