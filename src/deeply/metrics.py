import tensorflow.keras.backend as K

# https://github.com/keras-team/keras/issues/3611#issuecomment-243108708
def dice_coefficient(y_true, y_pred, smooth = 1):
    dtype     = y_pred.dtype

    y_true    = K.cast(y_true, dtype)

    axis      = (1, 2, 3)
    intersect = K.sum(y_true * y_pred, axis = axis)
    union     = K.sum(y_true, axis = axis) + K.sum(y_pred, axis = axis)

    return K.mean((2.0 * intersect + smooth) / (union + smooth), axis = 0)