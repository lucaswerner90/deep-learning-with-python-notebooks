import keras
from keras import ops
import numpy as np
import tensorflow as tf


class MeanAbsoluteErrorTracker(keras.metrics.Metric):
    def __init__(self, name="mean_absolute_error_tracker", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mae_sum = self.add_weight(name="mae_sum", initializer="zeros")
        self.num_samples = self.add_weight(
            name="num_samples", initializer="zeros", dtype="int64"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, dtype=self.dtype)
        y_pred = ops.cast(y_pred, dtype=self.dtype)

        mae = ops.abs(y_true - y_pred)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, dtype=self.dtype)
            mae = ops.multiply(mae, sample_weight)

        self.mae_sum.assign_add(ops.sum(mae))

        num_samples = ops.cast(ops.shape(y_true)[0], dtype=tf.int64)

        self.num_samples.assign_add(num_samples)

    def result(self):
        return self.mae_sum / ops.cast(self.num_samples, dtype=self.dtype)

    def reset_state(self):
        self.mae_sum.assign(0.0)
        self.num_samples.assign(0)


class TestMeanAbsoluteErrorTracker(tf.test.TestCase):
    def test_mae_calculation(self):
        mae_tracker = MeanAbsoluteErrorTracker()
        y_true = tf.constant([1.0, 2.0, 3.0, 4.0])
        y_pred = tf.constant([1.5, 2.5, 3.5, 4.5])

        mae_tracker.update_state(y_true, y_pred)

        self.assertAllClose(0.5, mae_tracker.result())

    def test_mae_reset(self):
        mae_tracker = MeanAbsoluteErrorTracker()
        y_true = tf.constant([1.0, 2.0, 3.0, 4.0])
        y_pred = tf.constant([1.5, 2.5, 3.5, 4.5])

        mae_tracker.update_state(y_true, y_pred)

        self.assertAllClose(0.5, mae_tracker.result())

        mae_tracker.reset_state()

        self.assertAllClose(0.0, mae_tracker.result())

    def test_mae_with_sample_weights(self):
        mae_tracker = MeanAbsoluteErrorTracker()
        y_true = tf.constant([1.0, 2.0, 3.0, 4.0])
        y_pred = tf.constant([1.5, 2.5, 3.5, 4.5])

        sample_weight = tf.constant([1.0, 0.5, 0.25, 0.0])

        mae_tracker.update_state(y_true, y_pred, sample_weight=sample_weight)

        self.assertAllClose(
            ((0.5 * 1.0) + (0.5 * 0.5) + (0.5 * 0.25) + (0.5 * 0.0)) / 4,
            mae_tracker.result(),
        )


if __name__ == "__main__":
    tf.test.main()
