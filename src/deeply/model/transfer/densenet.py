from deeply.model.dam import fully_connected_block
from tensorflow.keras import (
    Input
)
from tensorflow.keras.layers import (
    Layer,
    Flatten,
    Dense,
    Activation,
    BatchNormalization,
    Dropout
)

from deeply.model.base import BaseModel
from bpyutils.util.imports import import_handler
from deeply.const import DEFAULT

def fully_connected_block(x, units = 256, depth = 3, growth_rate = 0.5, activation = "relu",
    final_activation = "softmax", n_classes = 1, batch_norm = DEFAULT["batch_norm"],
    dropout_rate = DEFAULT["dropout_rate"], kernel_initializer = None):
    x = Flatten()(x)

    if batch_norm:
        x = BatchNormalization()(x)

    for _ in range(depth):
        x = Dense(units, kernel_initializer = kernel_initializer)(x)
        
        if batch_norm:
            x = BatchNormalization()(x)

        x = Activation(activation = activation)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        units = int(units * growth_rate)

    x = Dense(n_classes, activation = final_activation,
            kernel_initializer = kernel_initializer)(x)

    return x

def _transfer_densenet(module, *args, **kwargs):
    input_shape = kwargs.pop("input_shape", DEFAULT["transfer_densenet_input_shape"])
    weights     = kwargs.pop("weights",     DEFAULT["transfer_densenet_weights"])
    include_top = kwargs.pop("include_top", False)
    classes     = kwargs.pop("classes",     DEFAULT["transfer_densenet_classes"])

    DenseNet = import_handler(module)
    densenet = DenseNet(*args, **kwargs)
    densenet.trainable = True

    for layer in densenet.layers:
        if "conv5" in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    input_ = Input(input_shape = input_shape)
    model  = densenet(inputs = input_)

    output = fully_connected_block(model)

    model  = BaseModel(inputs = input_, outputs = output, name = "transfer-%s" % model.name)

    return model

def DenseNet121(*args, **kwargs):
    _transfer_densenet("tensorflow.keras.applications.DenseNet121", *args, **kwargs)

def DenseNet169(*args, **kwargs):
    _transfer_densenet("tensorflow.keras.applications.DenseNet169", *args, **kwargs)

def DenseNet201(*args, **kwargs):
    _transfer_densenet("tensorflow.keras.applications.DenseNet201", *args, **kwargs)