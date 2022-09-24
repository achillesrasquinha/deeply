from tensorflow.keras import (
    Sequential
)
from tensorflow.keras.layers import (
    InputLayer,
    Dense,
    Activation
)

def MLP(
    input_shape  = (1,),
    hidden_layer = (100,),
    n_output     = 1,
    activation   = "relu",
    final_activation = "softmax"
):
    model = Sequential()
    
    model.add(InputLayer(input_shape = input_shape))

    for layer in hidden_layer:
        model.add(Dense(layer))
        model.add(Activation(activation = activation))

    model.add(Dense(n_output))
    model.add(Activation(activation = final_activation))

    model.compile(
        optimizer = "sgd",
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"]
    )

    return model