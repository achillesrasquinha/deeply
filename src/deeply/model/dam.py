from tensorflow.keras.applications import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    InceptionResNetV2
)

from deeply.model.densenet import DenseNet161
from deeply.ensemble import Stacking

def DAM(models = None):
    """
    Deep AUC Maximzation
    """
    models  = models or [
        DenseNet121(),
        DenseNet169(),
        DenseNet201(),
        InceptionResNetV2()
    ]

    stacked = Stacking(models)

    return stacked