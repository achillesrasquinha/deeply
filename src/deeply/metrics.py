import tensorflow as tf
from   tensorflow.math import (
    reduce_sum,
    reduce_mean
)
import tensorflow.keras.backend as K

def y_cast(y_true, y_pred, dtype = K.floatx()):
    y_true = K.cast(y_true, dtype)
    y_pred = K.cast(y_pred, dtype)

    return (y_true, y_pred)

def jaccard_index(y_true, y_pred, smooth = 1):
    y_true, y_pred = y_cast(y_true, y_pred)
    
    intersect = reduce_sum(y_true * y_pred)
    sum_      = reduce_sum(K.abs(y_true)) + reduce_sum(K.abs(y_pred))

    jaccard   = reduce_mean((intersect + smooth) / (sum_ - intersect + smooth))

    return jaccard

def dice_coefficient(y_true, y_pred, smooth = 1):
    y_true, y_pred = y_cast(y_true, y_pred)

    intersect = reduce_sum(y_true * y_pred)
    union     = reduce_sum(y_true) + reduce_sum(y_pred)

    return reduce_mean((2.0 * intersect + smooth) / (union + smooth))

def tversky_index(y_true, y_pred, smooth = 1, alpha = 0.7):
    """
    Links
        1. https://arxiv.org/pdf/1810.07842.pdf
    """
    axis = (1, 2, 3)
    tp   = K.sum(y_true * y_pred, axis = axis)
    fn   = K.sum(y_true * (1 - y_pred), axis = axis)
    fp   = K.sum((1 - y_true) * y_pred, axis = axis)

    return K.mean((tp + smooth) / (tp + alpha * fn + (1 - alpha) * fp + smooth), axis = 0)

def focal_tversky_index(*args, **kwargs):
    """
    Links
        1. https://arxiv.org/pdf/1810.07842.pdf
    """
    tversky = tversky_index(*args, **kwargs) 
    gamma   = kwargs.get("gamma", 4/3)

    return K.pow(1 - tversky, gamma)