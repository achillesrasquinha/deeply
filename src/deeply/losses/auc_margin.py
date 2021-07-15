from tensorflow.keras.losses import (
    Loss
)
import tensorflow as tf

from deeply.const import DEFAULT
from deeply.losses.util import preprocess_y

class AUCMarginLoss(Loss):
    """
    Computes the AUC Margin Loss between labels and predictions.

    Binary/Cateogorical?

    :param y_true: true label.
    :param y_pred: predicted value.
    :param margin: 
    :param imratio: Imbalance Ratio.
    :param alpha:

    Example:
        >>> aucm = deeply.losses.AUCMarginLoss()
        >>> aucm(y_true, y_pred).numpy()

    References:
        Yuan, Zhuoning, et al. “Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification.” ArXiv:2012.03173 [Cs, Math, Stat], Dec. 2020. arXiv.org, http://arxiv.org/abs/2012.03173.
    """
    def __init__(self,
        margin  = DEFAULT["auc_margin_loss_margin"],
        imratio = DEFAULT["auc_margin_loss_imratio"],
        alpha   = DEFAULT["auc_margin_loss_alpha"], 
        axis    = -1,
        name    = "auc_margin_loss",
    ):
        super(AUCMarginLoss, self).__init__(
            auc_margin_loss,
            name    = name,
            margin  = margin,
            imratio = imratio,
            alpha   = alpha,
            axis    = axis
        )

def count(arr, value, dtype = tf.int32):
    return tf.reduce_sum(tf.cast(tf.equal(arr, value), dtype))

def auc_margin_loss(y_true, y_pred,
    margin  = DEFAULT["auc_margin_loss_margin"],
    imratio = DEFAULT["auc_margin_loss_imratio"],
    alpha   = DEFAULT["auc_margin_loss_alpha"],
    axis    = -1
):
    """
    Computes the AUC Margin Loss between labels and predictions.

    Example:
        >>> loss = deeply.losses.auc_margin_loss(y_true, y_pred)
        >>> loss.numpy()
    """
    y_true, y_pred = preprocess_y(y_true, y_pred)
    p = imratio
    
    if p is None:
        p = count(y_true, 1) / y_true.shape[0]

    n = (1 - p) # number of negative samples
    t = p * n

    n_p = tf.equal(y_true, 1)
    n_n = tf.equal(y_true, 0)

    foo = tf.constant(0)
    
    loss = \
          n * tf.reduce_mean((y_pred - foo) ** 2 * n_p, axis = axis) \
        + p * tf.reduce_mean((y_pred - foo) ** 2 * n_n, axis = axis) \
        + 2 * alpha * ( t * margin
            + tf.reduce_mean( p * y_pred * n_n - n * y_pred * n_p , axis = axis) ) \
                - t * (alpha ** 2)

    return loss