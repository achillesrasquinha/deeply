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

    "weights": "imagenet",

    "gan_learning_rate": 1e-4,
    "gan_plot_callback_samples": 16,
    "gan_discriminator_train_steps_offset": 3,
    "gan_gradient_penalty_weight": 10,

    "base_model_learning_rate": 1e-4,

    "generative_model_encoder_loss": "binary_crossentropy",
    "generative_model_decoder_loss": "binary_crossentropy",
    
    "generative_model_encoder_learning_rate": 1e-4,
    "generative_model_decoder_learning_rate": 1e-4,
}