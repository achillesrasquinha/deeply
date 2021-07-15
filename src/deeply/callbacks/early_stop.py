from tensorflow.keras.callbacks import EarlyStopping

from deeply.const import DEFAULT
from deeply.log import get_logger

logger = get_logger()

class GeneralizedEarlyStopping(EarlyStopping):
    """
    Early Stopping using Generalization Error.

    References
        [1]. Prechelt, Lutz. “Early Stopping — But When?” Neural Networks: Tricks of the Trade: Second Edition, edited by Grégoire Montavon et al., Springer, 2012, pp. 53–67. Springer Link, doi:10.1007/978-3-642-35289-8_5.
    """
    def __init__(self, *args, **kwargs):
        self._super = super()

        default = DEFAULT["generalized_early_stopping_monitor"]
        monitor = kwargs.get("monitor")
        if monitor not in ("gen_val_loss", "progress_quotient", ""):
            logger.warn("GeneralizedEarlyStopping monitor type %s is not found. Using default %s" % 
                (monitor, default))
            monitor = default

        baseline = kwargs.get("baseline", None)

        if monitor in ("gen_val_loss", "progress_quotient") and not baseline:
            raise TypeError("No limit provided.")

        kwargs["monitor"]  = monitor
        kwargs["mode"]     = "min"
        kwargs["patience"] = kwargs.get("patience", 1)

        self._super.__init__(*args, **kwargs)

        self.losses       = [ ]
        self.min_val_loss = None

    def on_train_begin(self, logs = None):
        self._super.on_train_begin(logs)

        self.limit        = None

        self.losses       = [ ]
        self.min_val_loss = None

    def on_epoch_end(self, epoch, logs = None):
        logs = logs or { }

        loss = logs.get("loss", None)
        val_loss = logs.get("val_loss", None)

        if val_loss is not None:
            if self.min_val_loss is not None:
                self.min_val_loss = min(self.min_val_loss, val_loss)
            else:
                self.min_val_loss = val_loss

            gen_val_loss = (val_loss / self.min_val_loss) - 1

            if loss is not None:
                self.losses.append(loss)

                training_progress = (sum(self.losses) / epoch * min(self.losses)) - 1
                logs["progress_quotient"] = (gen_val_loss / training_progress) * 0.1

            logs["gen_val_loss"] = gen_val_loss

        self._super.on_epoch_end(epoch, logs)