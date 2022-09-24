from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

from deeply.const import DEFAULT
from deeply.util.model import get_fit_args_kwargs

class BaseModel(Model):
    def __init__(self, *args, **kwargs):
        self._scaler = kwargs.pop("scaler", None)

        super_ = super(BaseModel, self)
        super_.__init__(*args, **kwargs)

        self.callbacks = []

    @property
    def scaler(self):
        return getattr(self, "_scaler", None)

    def plot(self, *args, **kwargs):
        kwargs["show_shapes"] = kwargs.get("show_shapes", True)
        return plot_model(self, *args, **kwargs)

    def add_callback(self, callback, *args, **kwargs):
        self.callbacks.append(callback)

    def fit(self, *args, **kwargs):
        args, kwargs = get_fit_args_kwargs(self, args, kwargs, custom = {
            "callbacks": self.callbacks
        })

        super_ = super(BaseModel, self)
        return super_.fit(*args, **kwargs)

    def compile(self, *args, **kwargs):
        learning_rate       = kwargs.pop("learning_rate", DEFAULT["base_model_learning_rate"])
        kwargs["optimizer"] = kwargs.get("optimizer", Adam(learning_rate = learning_rate))
        
        super_ = super(BaseModel, self)
        return super_.compile(*args, **kwargs)