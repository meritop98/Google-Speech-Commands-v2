import tensorflow as tf

class DataNormalization(tf.keras.layers.Layer):
    # implement __init__(), call, build functions for this layer
    # it should normalize the data with using mean and variance, which have to be updated for every train step
    def __init__(self, **kwargs):
        super(DataNormalization, self).__init__(**kwargs)
        self.mean = tf.Variable(initial_value=0.0, trainable=False)
        self.variance = tf.Variable(initial_value=1.0, trainable=False)

    def call(self, inputs, training=None, **kwargs):
        if training:
            # Update mean and variance based on the current batch
            batch_mean, batch_variance = tf.nn.moments(inputs, axes=list(
                range(len(inputs.shape))))
            self.mean.assign(batch_mean)
            self.variance.assign(batch_variance)
        return (inputs - self.mean) / tf.sqrt(self.variance + tf.keras.backend.epsilon())

class FeatureLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(FeatureLayer, self).__init__(**kwargs)
        self.axis = axis
    # optional, bonus points
    # implement getting the feature as non trainable layer 
    def call(self, inputs):
        # Compute mean and standard deviation as features
        mean = tf.math.reduce_mean(inputs, axis=self.axis)
        stddev = tf.math.reduce_std(inputs, axis=self.axis)
        return tf.concat([mean, stddev], axis=self.axis)