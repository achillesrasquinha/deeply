import tensorflow.keras.backend as K

def dice_coefficient(y_true, y_pred, smooth = 1):
    y_true_f  = K.flatten(y_true)
    y_pred_f  = K.flatten(y_pred)

    intersect = K.sum(y_true_f * y_pred_f)

    return (2.0 * intersect + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(*args, **kwargs):
    return 1 - dice_coefficient(*args, **kwargs)