import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        f1 = 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
        return tf.cast(f1, tf.float32)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def get_config(self):
        base_config = super().get_config()
        return base_config
