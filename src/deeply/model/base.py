from genericpath import exists
import os.path as osp

from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

from deeply.config import PATH
from deeply.const  import DEFAULT

class BaseModel(Model):
    # def __init__(self, *args, **kwargs):
    #     self.super = super(BaseModel, self)
    #     self.super.__init__(*args, **kwargs)

    # def compile(self, *args, **kwargs):
    #     kwargs["optimizer"] = kwargs.get("optimizer", DEFAULT["optimizer"])
    #     kwargs["loss"]      = kwargs.get("loss",      DEFAULT["loss"])

    #     metrics = kwargs.get("metrics",   [])

    #     if "accuracy" not in metrics:
    #         metrics.append("accuracy")

    #     kwargs["metrics"]   = metrics
    
    #     return self._super.compile(*args, **kwargs)

    def plot(self, *args, **kwargs):
        return plot_model(self, *args, **kwargs)

    # def fit(self, *args, **kwargs):
        # log_dir  = osp.join(PATH["CACHE"], "tensorboard-logs")
        # makedirs(log_dir, exist_ok = True)
        # log_path = osp.join(log_dir, get_random_str())

        # tb_cb    = TensorBoard(log_dir = log_path)

        # callbacks = list(kwargs.pop("callbacks", []))
        # callbacks.append(tb_cb)

        # kwargs["callbacks"] = [tb_cb]

        # return self.super.fit(*args, **kwargs)