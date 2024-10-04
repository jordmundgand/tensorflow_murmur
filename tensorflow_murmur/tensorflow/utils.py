import tensorflow as tf

def get_strategy(platform='kaggle', device='TPU'):
    if (platform=='kaggle') and (device=='TPU'):
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.tpu.experimental.initialize_tpu_system(tpu)
        return tf.distribute.TPUStrategy(tpu)
    elif (platform=='kaggle') and (device=='multyGPU'):
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        return strategy
    elif (platform=='colab') and (device=='TPU'): 
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        return tf.distribute.TPUStrategy(tpu)
