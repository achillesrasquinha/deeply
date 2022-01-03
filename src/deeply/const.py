import multiprocessing as mp

from bpyutils.util.environ import getenv

from deeply.__attr__ import __name__ as NAME

_PREFIX = NAME.upper()

N_JOBS  = getenv("JOBS", mp.cpu_count(), prefix = _PREFIX)

DEFAULT = \
{
    "batch_norm": True,
    "dropout_rate": 0.2,

    "batch_size": 32,
    "loss": "categorical_crossentropy",
    "optimizer": "adam",

    "initial_filters": 64,
    "batch_normalization": True,
    "dropout_rate": 0.4,

    "generalized_early_stopping_monitor": "progress_quotient",

    "auc_margin_loss_margin": 1.0,
    "auc_margin_loss_imratio": None,
    "auc_margin_loss_alpha": 0,

    "transfer_densenet_input_shape": (224, 224, 3),
    "transfer_densenet_weights": "imagenet",
    "transfer_densenet_classes": 1,
    
    "densenet_growth_rate": 32,

    "weights": "imagenet"
}