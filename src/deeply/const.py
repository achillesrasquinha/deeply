DEFAULT = \
{
    "batch_size": 32,
    "loss": "categorical_crossentropy",
    "optimizer": "adam",

    "initial_filters": 64,
    "batch_normalization": True,
    "dropout_rate": 0.4,

    "generalized_early_stopping_monitor": "progress_quotient",

    "auc_margin_loss_margin": 1.0,
    "auc_margin_loss_imratio": None,
    "auc_margin_loss_alpha": 0
}