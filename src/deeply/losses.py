from deeply.metrics import (
    dice_coefficient,
    tversky_index
)

# https://github.com/keras-team/keras/issues/3611#issuecomment-246305119
def dice_loss(*args, **kwargs):
    return 1 - dice_coefficient(*args, **kwargs)

def tversky_loss(*args, **kwargs):
    return 1 - tversky_index(*args, **kwargs)

def auc_margin_loss(y_true, y_pred, margin = 1.0, alpha = 0):
    """
    AUC margin loss

    References:
        [1] Yuan, Zhuoning, et al. “Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification.” ArXiv:2012.03173 [Cs, Math, Stat], Dec. 2020. arXiv.org, http://arxiv.org/abs/2012.03173.
    """
    # TODO: implement.
    pass
    # 2 * alpha * () - (alpha ** 2)