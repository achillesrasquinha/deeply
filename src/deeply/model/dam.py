from tensorflow.keras import (
    Input
)
from tensorflow.keras.applications import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    InceptionResNetV2
)

from deeply.model.base import BaseModel
# from deeply.model.densenet import DenseNet161
from deeply.ensemble import Stacking
from deeply.losses import auc_margin_loss

class DAMModel(BaseModel):
    def __init__(self, *args, **kwargs):
        self._super = super(DAMModel, self)
        self._super.__init__(*args, **kwargs)

    def compile(self, *args, **kwargs):
        kwargs["loss"] = auc_margin_loss

        return self._super.compile(*args, **kwargs)

def DAM(
    models      = None,
    input_shape = (320, 320, 1)
):
    """
    Deep AUC Maximzation
    """
    if not models:
        input_ = Input(shape = input_shape)
        nets   = [DenseNet121, DenseNet169, DenseNet201, InceptionResNetV2]

        models = [x(input_tensor = input_) for x in nets]

    stacked = Stacking(models)

    return DAMModel(stacked)