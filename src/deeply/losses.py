from deeply.metrics import dice_coefficient

# https://github.com/keras-team/keras/issues/3611#issuecomment-246305119
def dice_loss(*args, **kwargs):
    return 1 - dice_coefficient(*args, **kwargs)