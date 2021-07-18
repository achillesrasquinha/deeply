import os.path as osp

from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

from deeply.callbacks import ProgressStepCallback
class BaseModel(Model):
    def __init__(self, *args, **kwargs):
        self._super = super(BaseModel, self)
        self._super.__init__(*args, **kwargs)

    def plot(self, *args, **kwargs):
        return plot_model(self, *args, **kwargs)

    def fit(self, *args, **kwargs):
        callbacks = list(kwargs.pop("callbacks", []))
        callbacks.append(ProgressStepCallback())

        kwargs["callbacks"] = callbacks

        return self._super.fit(*args, **kwargs)