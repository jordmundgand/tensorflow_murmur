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

