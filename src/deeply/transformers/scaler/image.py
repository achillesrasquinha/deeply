import tensorflow as tf
from deeply.transformers.scaler.base import BaseScaler

class ImageScaler(BaseScaler):
    def fit_transform(self, X):
        return (tf.cast(X, tf.float32) - 127.5) / 127.5

    def inverse_transform(self, X):
        return (tf.cast(X, tf.float32) * 127.5) + 127.5