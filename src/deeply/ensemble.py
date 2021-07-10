from tensorflow.keras.layers import (
    Layer,
    Concatenate
)

from deeply.model.base import BaseModel as Model

class EnsembleBlock(Layer):
    def __init__(self, layer, *args, **kwargs):
        self.super = super(EnsembleBlock, self)
        self.layer = layer

    def call(self, x, training = False):
        return self.layer(training = training)(x)

def Stacking(models,
    name = "stacking"
):
    # rename model layers
    for model in models:
        for layer in model.layers:
            layer._name = "%s_%s_%s" % (name, model.name, layer.name)

    inputs, outputs = list(zip(*[( EnsembleBlock(model.input), model.output ) for model in models]))

    output = Concatenate()(outputs)
    
    model  = Model(inputs = inputs, outputs = output, name = name)
    
    return model