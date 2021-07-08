from tensorflow.keras.layers import (
    Layer,
    Concatenate
)

from deeply.model.base import BaseModel as Model

def Stacking(models,
    name = "stacking"
):
    # rename model layers
    for model in models:
        for layer in model.layers:
            layer._name = "%s_%s_%s" % (name, model.name, layer.name)

    inputs, outputs = list(zip(*[( model.input, model.output ) for model in models]))
    output = Concatenate()(outputs)
    
    model  = Model(inputs = inputs, outputs = output, name = name)
    
    return model