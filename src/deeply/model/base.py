import os.path as osp

from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

from tqdm.keras import TqdmCallback

from deeply.callbacks import ProgressStepCallback
class BaseModel(Model):
    def __init__(self, *args, **kwargs):
        self._super = super(BaseModel, self)
        self._super.__init__(*args, **kwargs)

    def plot(self, *args, **kwargs):
        kwargs["show_shapes"] = kwargs.get("show_shapes", True)
        
        return plot_model(self, *args, **kwargs)

    def fit(self, *args, **kwargs):
        verbose   = kwargs.pop("verbose", 0)
        
        callbacks = list(kwargs.pop("callbacks", []))
        callbacks.append(ProgressStepCallback())

        callbacks.append(TqdmCallback(verbose = verbose))

        kwargs["callbacks"] = callbacks

        return self._super.fit(*args, **kwargs)