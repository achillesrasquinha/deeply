from deeply.metrics import dice_coefficient

def dice_loss(*args, **kwargs):
    return 1 - dice_coefficient(*args, **kwargs)