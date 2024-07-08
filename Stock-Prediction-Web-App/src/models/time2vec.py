import tensorflow as tf
import tensorflow.keras as keras

class Time2Vec(keras.layers.Layer):
    def __init__(self, kernel_size):
        super().__init__(trainable=True, name='Time2Vec')
        self.k = kernel_size
    def build(self, input_shape):
        self.w = self.add_weight(name='w',shape=(input_shape[1], 1),initializer='uniform',trainable=True)
        self.p = self.add_weight(name='p',shape=(input_shape[1], 1),initializer='uniform',trainable=True)
        self.W = self.add_weight(name='W',shape=(input_shape[-1], self.k),initializer='uniform',trainable=True)
        self.P = self.add_weight(name='P',shape=(input_shape[1], self.k),initializer='uniform',trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # print(inputs.shape , self.w.shape, self.p.shape, self.W.shape, self.P.shape)
        original = self.w * inputs + self.p
        sin_trans = tf.math.sin(tf.tensordot(original, self.W , axes=1) + self.P)
        # print(original.shape , sin_trans.shape)
        # ans = tf.concat([sin_trans, original], -1)
        return sin_trans