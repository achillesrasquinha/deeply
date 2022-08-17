from bpyutils.util._dict import merge_dict
from bpyutils._compat import iteritems, iterkeys

from tensorflow.keras.layers import (
    Layer,
    Activation,
    Dropout,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense
)

class ActivationBatchNormDropout(Layer):
    """
        Activation -> Batch Normalization -> Dropout
    """
    def __init__(self, activation = "relu", batch_norm = True, dropout_rate = 0.5, *args, **kwargs):
        self._super = super(ActivationBatchNormDropout, self)
        self._super.__init__(*args, **kwargs)

        if not isinstance(activation, Layer):
            self.activation = Activation(activation = activation) if activation else None
        else:
            self.activation = activation

        self.batch_norm  = BatchNormalization() if batch_norm else None
        self.dropout     = Dropout(rate = dropout_rate) if dropout_rate else None

    def call(self, inputs, training = False):
        x = inputs

        if training and self.batch_norm:
            x = self.batch_norm(x)

        if self.activation:
            x = self.activation(x)

        # https://stats.stackexchange.com/a/317313
        if training and self.dropout:
            x = self.dropout(x)

        return x

def get_default_block_kwargs():
    return {
        "activation": "relu",
        "width": 2,
        "batch_norm": True,
        "dropout_rate": 0.2,
        "kernel_initializer": None,
        "use_bias": True
    }

def generate_block(name, layer_type, layer_kwargs = {}):
    default_kwargs = get_default_block_kwargs()

    merged_kwargs  = merge_dict(default_kwargs, layer_kwargs)

    class Block(Layer):
        def __init__(self, *args, **kwargs):
            args = list(args)

            if len(args):
                setattr(self, "units", args.pop(0))
            else:
                raise ValueError("No units provided.")

            block_kwargs = { }
            final_layer_kwargs = layer_kwargs

            for key, default_value in iteritems(merged_kwargs):
                if key in kwargs:
                    block_kwargs[key] = kwargs.pop(key, default_value)
                    setattr(self, key, block_kwargs[key])

                    if key in layer_kwargs:
                        final_layer_kwargs[key] = block_kwargs[key]

            self._super = super(Block, self)
            self._super.__init__(*args, **kwargs)

            self.layers       = [ ]
            self.activations  = [ ]

            for _ in range(self.width):
                layer = layer_type(self.units, kernel_initializer = self.kernel_initializer,
                    **final_layer_kwargs)
                self.layers.append(layer)

                activation = ActivationBatchNormDropout(activation = self.activation,
                    batch_norm = self.batch_norm, dropout_rate = self.dropout_rate)
                self.activations.append(activation)

            self._block_attrs = merge_dict(block_kwargs, { "units": self.units })

        def call(self, inputs, training = False):
            x = inputs

            for i in range(self.width):
                x = self.layers[i](x, training = training)
                x = self.activations[i](x, training = training)

            return x

        def get_config(self):
            config = self._super.get_config()

            for key in iterkeys(self._block_attrs):
                config.update({ key: getattr(self, key)})

            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    Block.__name__ = name

    return Block

def get_conv_block_kwargs():
    return {
        "kernel_size": 3,
        "strides": 1,
        "padding": "valid"
    }

DenseBlock = generate_block(
    name = "DenseBlock",
    layer_type = Dense
)
ConvBlock  = generate_block(
    name = "ConvBlock",
    layer_type   = Conv2D,
    layer_kwargs = get_conv_block_kwargs()
)
Conv2DTransposeBlock = generate_block(
    name = "Conv2DTransposeBlock",
    layer_type   = Conv2DTranspose,
    layer_kwargs = get_conv_block_kwargs()
)