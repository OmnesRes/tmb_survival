import tensorflow as tf

class Encoders:
    class Encoder:
        def __init__(self, shape=None, layers=(2,), dropout=0):
            self.shape = shape
            self.model = None
            self.layers = layers
            self.dropout = dropout
            self.build()

        def build(self, *args, **kwarg):
            input = tf.keras.layers.Input(self.shape, dtype=tf.float32)
            hidden = [input]
            for i in self.layers:
                hidden.append(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.softplus)(hidden[-1]))
                if self.dropout:
                    hidden.append(tf.keras.layers.Dropout(self.dropout)(hidden[-1]))
            self.model = tf.keras.Model(inputs=[input], outputs=[hidden[-1]])

class NN:
    def __init__(self, encoders=[], layers=(16, 8, 4), default_activation=tf.keras.activations.softplus, norm=False):
        self.encoders = encoders
        self.layers = layers
        self.norm = norm
        self.default_activation = default_activation
        self.model = None
        self.build()

    def build(self):
        inputs = [[tf.keras.layers.Input(shape=input_tensor.shape[1:], dtype=input_tensor.dtype) for input_tensor in encoder.inputs] for encoder in self.encoders]
        encodings = [encoder(input) for input, encoder in zip(inputs, self.encoders)]
        fused = [tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(encodings)]
        for index, i in enumerate(self.layers):
            fused.append(tf.keras.layers.Dense(units=i, activation=self.default_activation)(fused[-1]))
        output = tf.keras.layers.Dense(units=1, activation=None, use_bias=False)(fused[-1])
        if self.norm:
            output = tf.keras.layers.BatchNormalization(momentum=0)(output)
        self.model = tf.keras.Model(inputs=inputs, outputs=[output])




