from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

from deeply.const import DEFAULT
from deeply.util.model import get_fit_kwargs

class BaseModel(Model):
    def __init__(self, *args, **kwargs):
        self._super = super(BaseModel, self)
        self._super.__init__(**kwargs)

        self.callbacks = []

    def plot(self, *args, **kwargs):
        kwargs["show_shapes"] = kwargs.get("show_shapes", True)
        return plot_model(self, *args, **kwargs)

    def add_callback(self, callback, *args, **kwargs):
        self.callbacks.append(callback)

    def fit(self, *args, **kwargs):
        kwargs = get_fit_kwargs(self, kwargs, custom = {
            "callbacks": self.callbacks
        })

        return self._super.fit(*args, **kwargs)

    def compile(self, *args, **kwargs):
        learning_rate       = kwargs.pop("learning_rate", DEFAULT["base_model_learning_rate"])
        kwargs["optimizer"] = kwargs.get("optimizer", Adam(learning_rate = learning_rate))
        return self._super.compile(*args, **kwargs)