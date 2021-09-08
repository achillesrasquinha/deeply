from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Layer,
    GlobalAveragePooling2D,
    Reshape,
    Dense,
    Multiply
)

def squeeze_excitation_block(inputs, ratio = 16, activation = "relu", final_activation = "sigmoid", kernel_initializer = None):
    x = inputs
    # print(dir(x))
    print(x._shape_val)
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = x._shape_val[channel_axis]
    shape   = (1, 1, filters)

    x = GlobalAveragePooling2D()(x)
    x = Reshape(shape)(x)
    x = Dense(filters // ratio, activation = activation, kernel_initialier = kernel_initializer, use_bias = False)(x)
    x = Dense(filters, activation = final_activation, kernel_initializer = kernel_initializer, use_bias = False)(x)

    x = Multiply()([x, inputs])

    return x