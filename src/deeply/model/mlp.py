from tensorflow.keras import (
    Sequential
)
from tensorflow.keras.layers import (
    Input,
    Dense,
    Activation
)
from tensorflow.keras.optimizers import (
    Adam
)

from deeply.model.base import (
    BaseModel
)
from deeply.model.layer import (
    DenseBlock
)
from deeply.metrics import r2_score

def MLP(
    input_shape   = (1,),
    hidden        = (100,),
    n_output      = 1,
    activation    = "relu",
    final_activation = "softmax",
    optimizer     = Adam,
    loss          = "category_crossentropy",
    layer_block   = DenseBlock,
    batch_norm    = False,
    dropout_rate  = 0.3,
    width         = 1,
    kernel_initializer = "glorot_uniform",
    learning_rate = 0.001,
    metrics       = ["accuracy"],
    name          = "mlp",
    weights       = None,
    **kwargs
):
    input_ = Input(shape = input_shape)
    m = input_

    layer_args = { "batch_norm": batch_norm, "dropout_rate": dropout_rate, "width": width,
        "activation": activation, "kernel_initializer": kernel_initializer }

    for layer in hidden:
        m = DenseBlock(layer, **layer_args)(m)

    layer_args["batch_norm"]   = False
    layer_args["dropout_rate"] = 0
    layer_args["activation"]   = None

    m = layer_block(n_output, **layer_args)(m)

    if final_activation:
        m = Activation(activation = final_activation)(m)
    
    model = BaseModel(inputs = [input_], outputs = [m],
        name = name or "mlp")

    model.compile(
        optimizer = optimizer(learning_rate = learning_rate),
        loss      = loss,
        metrics   = metrics
    )

    if weights:
        model.load_weights(weights, **kwargs.get("load_weights_kwargs", {}))

    return model

def MLPRegressor(*args, **kwargs):
    kwargs["hidden"]  = kwargs.get("hidden", (100,))
    kwargs["metrics"] = kwargs.get("metrics", []) + ["mae", "mse", "mape", r2_score]
    kwargs["final_activation"] = None
    kwargs["learning_rate"] = kwargs.get("learning_rate", 0.001)
    kwargs["loss"] = kwargs.get("loss", "mean_absolute_error")
    kwargs["name"] = kwargs.get("name", "mlpreg")

    return MLP(*args, **kwargs)