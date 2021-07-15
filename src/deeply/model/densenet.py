from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Layer,
    Conv2D,
    MaxPooling2D,
    Activation,
    BatchNormalization,
    Dropout
)

from deeply.model.base import BaseModel
from deeply.const import DEFAULT

class DenseBlock(Layer):
    pass

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

    References
        https://arxiv.org/pdf/1608.06993.pdf
    """
    input_ = Input(shape = input_shape)

    x = Conv2D(init_filters, 7, strides = 2)(input_)
    x = BatchNormalization()(x)
    x = Activation(activation = "relu")(x)

    if dropout_rate:
        x = Dropout(dropout_rate = dropout_rate)(x)

    x = MaxPooling2D(kernel_size = 3, strides = 2)(x)

    # for block in blocks:
    #     x = DenseBlock()(x)

    output_layer = x

    return DenseNetModel(inputs = input_, outputs = output_layer, name = name)

def DenseNet161(*args, **kwargs):
    """
    DenseNet-161
    """
    return DenseNet(*args, **kwargs)