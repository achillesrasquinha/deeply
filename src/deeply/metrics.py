import tensorflow.keras.backend as K

def dice_coefficient(y_true, y_pred, smooth = 1):
    dtype     = y_pred.dtype

    y_true_f  = K.flatten(K.cast(y_true, dtype))
    y_pred_f  = K.flatten(y_pred)

    intersect = K.sum(y_true_f * y_pred_f)

    return (2.0 * intersect + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)