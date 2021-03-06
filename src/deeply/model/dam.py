from tensorflow.keras import (
    Input
)
from tensorflow.keras.applications import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    InceptionResNetV2
)

from deeply.const import DEFAULT
from deeply.model.base import BaseModel
from deeply.model.densenet import DenseNet161
from deeply.ensemble import Stacking

def DAM(
    models       = None,
    n_classes    = 1,
    input_shape  = (320, 320, 1),
    weights      = "imagenet",
    name         = "dam",

    batch_norm   = DEFAULT["batch_norm"],
    dropout_rate = DEFAULT["dropout_rate"],

    fc_units        = 256,
    fc_depth        = 2,
    fc_growth_rate  = 0.5,
    fc_activation   = "relu"
):
    """
    Deep AUC Maximzation
    """
    if not models:
        nets   = [DenseNet121, DenseNet169, DenseNet201]#, InceptionResNetV2]
        models = [ ]

        for net in nets:
            net = net(weights = weights, include_top = False, classes = n_classes)
            net.trainable = True

            # freeze layers
            for layer in net.layers:
                if "conv5" in layer.name:
                    layer.trainable = True
                else:
                    layer.trainable = False

            input_  = Input(shape = input_shape)
            model   = net(inputs = input_)

            output  = fully_connected_block(model, units = fc_units, depth = fc_depth, growth_rate = fc_growth_rate,
                        batch_norm = batch_norm, dropout_rate = dropout_rate,
                        activation = fc_activation, n_classes = n_classes)

            model   = BaseModel(inputs = input_, outputs = output, name = net.name)

            models.append(model)

    stacked = Stacking(models)

    return stacked