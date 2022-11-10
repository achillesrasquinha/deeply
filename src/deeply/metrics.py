import tensorflow as tf
from   tensorflow.math import (
    reduce_sum,
    reduce_mean
)
import tensorflow.keras.backend as K
from tensorflow_addons.metrics import (
    RSquare
)
from sklearn.metrics import (
    r2_score as sk_r2_score
)

from bpyutils import log

from deeply.__attr__ import __name__ as NAME

logger   = log.get_logger(NAME)
_rsquare = RSquare()

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
    y_true, y_pred = y_cast(y_true, y_pred)

    tp = reduce_sum(y_true * y_pred)
    fn = reduce_sum(y_true * (1 - y_pred))
    fp = reduce_sum((1 - y_true) * y_pred)

    return reduce_mean((tp + smooth) / (tp + alpha * fn + (1 - alpha) * fp + smooth))

def focal_tversky_index(*args, **kwargs):
    """
    Links
        1. https://arxiv.org/pdf/1810.07842.pdf
    """
    tversky = tversky_index(*args, **kwargs) 
    gamma   = kwargs.get("gamma", 4/3)

    return K.pow(1 - tversky, gamma)

def r2_score(y_true, y_pred, *args, **kwargs):
    clip = kwargs.get("clip", False)

    y_true, y_pred = y_cast(y_true, y_pred)

    _rsquare.update_state(y_true, y_pred)

    result = _rsquare.result()

    if clip:
        result = K.clip(result, 0, 1)

    return result

def regression_report(model, X_test, Y_test):
    y_pred  = model.predict(X_test)
    metrics = [{
        "name": "mean-squared-error",
        "func": lambda y_true, y_pred: K.mean(tf.keras.metrics.mean_squared_error(y_true, y_pred)).numpy()
    }, {
        "name": "mean-absolute-error",
        "func": lambda y_true, y_pred: K.mean(tf.keras.metrics.mean_absolute_error(y_true, y_pred)).numpy()
    }, {
        "name": "mean-absolute-percent-error",
        "func": lambda y_true, y_pred: K.mean(tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)).numpy()
    },{
        "name": "r2-score",
        "func": lambda y_true, y_pred: r2_score(y_true, y_pred).numpy()
    }]

    report = {}

    logger.info("Regression Report:")

    for metric in metrics:
        report[metric["name"]] = metric["func"](Y_test, y_pred)
        logger.info("\t%s: %s" % (metric["name"], report[metric["name"]]))

    return report