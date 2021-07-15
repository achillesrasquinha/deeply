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
# from deeply.losses import auc_margin_loss

# class DAMModel(BaseModel):
#     def __init__(self, *args, **kwargs):
#         self._super = super(DAMModel, self)
#         self._super.__init__(*args, **kwargs)

    # def compile(self, *args, **kwargs):
    #     kwargs["loss"] = kwargs.get("loss", "binary_crossentropy")
    #     return self._super.compile(*args, **kwargs)

def DAM(
    models      = None,
    input_shape = (320, 320, 1),
    weights     = None,
    name        = "dam"
):
    """
    Deep AUC Maximzation
    """
    if not models:
        nets   = [DenseNet121, DenseNet169, DenseNet201, InceptionResNetV2]
        models = [x(input_shape = input_shape, weights = weights) for x in nets]

    stacked = Stacking(models)

    # model   = DAMModel(inputs = stacked.input, outputs = stacked.output, name = name)

    return stacked