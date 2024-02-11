import tensorflow as tf

def masked_accuracy(label, pred):
  '''Classical transformers masked SparseCategoricalAccuracy metric function.'''
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

def batch_average(label, pred):
  return tf.reduce_mean(pred)
