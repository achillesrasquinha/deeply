from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

from bpyutils.util.model import get_fit_kwargs
class BaseModel(Model):
    def __init__(self, *args, **kwargs):
        self._super = super(BaseModel, self)
        self._super.__init__(*args, **kwargs)

    def plot(self, *args, **kwargs):
        kwargs["show_shapes"] = kwargs.get("show_shapes", True)
        return plot_model(self, *args, **kwargs)

    def fit(self, *args, **kwargs):
        kwargs = get_fit_kwargs(self, kwargs)
        return self._super.fit(*args, **kwargs)