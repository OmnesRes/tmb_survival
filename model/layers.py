import tensorflow as tf

class Activations:
    class ASU(tf.keras.layers.Layer):
        def __init__(self, trainable=True, lower_asymptote=0., upper_asymptote=1., alpha_init=1., bias_init=None):
            super(Activations.ASU, self).__init__()
            self.trainable = trainable
            self.lower_asymptote = lower_asymptote
            self.upper_asymptote = upper_asymptote
            self.alpha_init = alpha_init
            self.lower_alpha = None
            self.upper_alpha = None
            self.bias_init = bias_init
            self.bias = None

        @staticmethod
        def activation_function(x, lower_asymptote, upper_asymptote, lower_alpha, upper_alpha):
            x_2 = x ** 2
            lower_sqrt = (lower_alpha + x_2) ** (1 / 2)
            upper_sqrt = (upper_alpha + x_2) ** (1 / 2)
            return lower_asymptote + ((upper_asymptote - lower_asymptote) * ((x + lower_sqrt) / (lower_sqrt + upper_sqrt)))

        def build(self, input_shape):
            self.lower_alpha = self.add_weight(shape=[input_shape[-1], ],
                                               initializer=tf.keras.initializers.constant(self.alpha_init),
                                               dtype=tf.float32, trainable=self.trainable)
            self.upper_alpha = self.add_weight(shape=[input_shape[-1], ],
                                               initializer=tf.keras.initializers.constant(self.alpha_init),
                                               dtype=tf.float32, trainable=self.trainable)
            if self.bias_init is not None:
                self.bias = self.add_weight(shape=[input_shape[-1], ], initializer=tf.keras.initializers.constant(self.alpha_init), dtype=tf.float32, trainable=self.trainable)


        def call(self, inputs, **kwargs):
            return self.activation_function(inputs + self.bias if self.bias is not None else inputs,
                                            lower_asymptote=self.lower_asymptote, upper_asymptote=self.upper_asymptote,
                                            lower_alpha=tf.exp(self.lower_alpha), upper_alpha=tf.exp(self.upper_alpha))


class Losses:
    class CoxPH(tf.keras.losses.Loss):
        def __init__(self, name='coxph', cancers=1):
            super(Losses.CoxPH, self).__init__(name=name)
            self.cancers = cancers

        def call(self, y_true, y_pred):
            total_losses = []
            for cancer in range(self.cancers):
                mask = tf.equal(y_true[:, -1], cancer)
                cancer_y_true = y_true[mask]
                cancer_y_pred = y_pred[mask]
                time_d = tf.cast(cancer_y_true[:, 0][tf.newaxis, :] <= cancer_y_true[:, 0][:, tf.newaxis], tf.float32)
                loss = (tf.math.log(tf.tensordot(time_d, tf.math.exp(cancer_y_pred[:, 0][:, tf.newaxis]), [0, 0])[:, 0]) - cancer_y_pred[:, 0]) * cancer_y_true[:, 1]
                total_losses.append(loss)
            return tf.concat(total_losses, axis=-1)

        def __call__(self, y_true, y_pred, sample_weight=None):
            ##sample weights out of order for multiple cancers, need to reweight based on events. Don't use weighting for now.
            losses = self.call(y_true, y_pred)
            if sample_weight is not None:
                return tf.reduce_sum(losses * sample_weight[:, 0]) / tf.reduce_sum(sample_weight)
            else:
                return tf.reduce_mean(losses)
    