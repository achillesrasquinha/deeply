import tensorflow as tf
import tensorflow.keras.backend as K

from deeply.losses.auc_margin import (
    AUCMarginLoss,
    auc_margin_loss
)

from deeply.metrics import (
    dice_coefficient,
    tversky_index,
    focal_tversky_index
)

# https://github.com/keras-team/keras/issues/3611#issuecomment-246305119
def dice_loss(*args, **kwargs):
    return 1 - dice_coefficient(*args, **kwargs)

def tversky_loss(*args, **kwargs):
    return 1 - tversky_index(*args, **kwargs)

def focal_tversky_loss(*args, **kwargs):
    return 1 - focal_tversky_index(*args, **kwargs)

def bernoulli_loss(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis = -1)