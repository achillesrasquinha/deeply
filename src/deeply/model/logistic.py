from tensorflow.keras import (
    Sequential
)
from tensorflow.keras.layers import (
    InputLayer,
    Dense,
    Activation
)

def LogisticRegression(
    input_shape = (1,),
    n_classes   = 1,
    activation  = "softmax"
):
    model = Sequential()
    
    model.add(InputLayer(input_shape = input_shape))
    model.add(Dense(n_classes))
    model.add(Activation(activation = activation))

    model.compile(
        optimizer = "sgd",
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"]
    )

    return model