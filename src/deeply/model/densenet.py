from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Conv2D
)

from deeply.model.base import BaseModel
from deeply.const import DEFAULT

class DenseNetModel(BaseModel):
    pass

def DenseNet(
    input_shape  = None,
    init_filters = DEFAULT["initial_filters"],
    batch_norm   = DEFAULT["batch_normalization"],
    dropout_rate = 0,
    name         = "densenet"
):
    """
    Densely Connected Convolutional Neural Networks.
    """
    input_ = Input(shape = input_shape)

    x = Conv2D(init_filters, 7, strides = 2)(input_)

    output_layer = x

    return DenseNetModel(inputs = input_, outputs = output_layer, name = name)

def DenseNet161(*args, **kwargs):
    """
    DenseNet-161
    """
    return DenseNet(*args, **kwargs)