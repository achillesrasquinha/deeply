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

        if self.activation:
            x = self.activation(x)
        
        if training and self.batch_norm:
            x = self.batch_norm(x)

        # https://stats.stackexchange.com/a/317313
        if training and self.dropout:
            x = self.dropout(x)

        return x

class DenseBlock(Layer):
    def __init__(self, units, activation = "relu", width = 2, batch_norm = True,
        dropout_rate = 0.2, kernel_initializer = None, *args, **kwargs):
        self._super = super(DenseBlock, self)
        self._super.__init__(*args, **kwargs)

        self.units        = units
        self.activation   = activation
        self.batch_norm   = batch_norm
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        self.denses       = [ ]
        self.activations  = [ ]

        for _ in range(width):
            dense = Dense(units, kernel_initializer = kernel_initializer)
            self.denses.append(dense)

            activation = ActivationBatchNormDropout(activation = activation,
                batch_norm = batch_norm, dropout_rate = dropout_rate)
            self.activations.append(activation)

        self.width        = width

    def call(self, inputs, training = False):
        x = inputs

        for i in range(self.width):
            x = self.denses[i](x, training = training)

            x = self.activations[i](x, training = training)

        return x

    def get_config(self):
        return {
            "units": self.units,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "dropout_rate": self.dropout_rate,
            "width": self.width,
            "kernel_initializer": self.kernel_initializer,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ConvBlock(Layer):
    def __init__(self, filters, kernel_size = 3, activation = "relu",
        width = 2, batch_norm = True, dropout_rate = 0.2, kernel_initializer = None,
        padding = "valid", strides = 1, *args, **kwargs):
        self._super = super(ConvBlock, self)
        self._super.__init__(*args, **kwargs)

        self.filters      = filters
        self.kernel_size  = kernel_size
        self.strides      = strides
        self.activation   = activation
        self.batch_norm   = batch_norm
        self.dropout_rate = dropout_rate
        self.padding      = padding
        self.kernel_initializer = kernel_initializer

        self.convs        = [ ]
        self.activations  = [ ]

        for _ in range(width):
            conv = Conv2D(filters = filters, kernel_size = kernel_size,
                kernel_initializer = kernel_initializer, padding = padding,
                strides = strides)
            self.convs.append(conv)

            activation = ActivationBatchNormDropout(activation = activation,
                batch_norm = batch_norm, dropout_rate = dropout_rate)
            self.activations.append(activation)

        self.width = width

    def call(self, inputs, training = False):
        x = inputs

        for i in range(self.width):
            x = self.convs[i](x)

            x = self.activations[i](x, training = training)

        return x

    def get_config(self):
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "activation": self.activation,
            "width": self.width,
            "batch_norm": self.batch_norm,
            "dropout_rate": self.dropout_rate,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Conv2DTransposeBlock(Layer):
    def __init__(self, filters, kernel_size = 3, activation = "relu",
        width = 1, batch_norm = True, dropout_rate = 0.2, kernel_initializer = None,
        padding = "valid", strides = 1, *args, **kwargs):
        self._super = super(Conv2DTransposeBlock, self)
        self._super.__init__(*args, **kwargs)

        self.filters      = filters
        self.kernel_size  = kernel_size
        self.strides      = strides
        self.activation   = activation
        self.batch_norm   = batch_norm
        self.dropout_rate = dropout_rate
        self.padding      = padding
        self.kernel_initializer = kernel_initializer

        self.convs        = [ ]
        self.activations  = [ ]

        for _ in range(width):
            conv = Conv2DTranspose(filters = filters, kernel_size = kernel_size,
                kernel_initializer = kernel_initializer, padding = padding,
                strides = strides)
            self.convs.append(conv)

            activation = ActivationBatchNormDropout(activation = activation,
                batch_norm = batch_norm, dropout_rate = dropout_rate)
            self.activations.append(activation)

        self.width = width

    def call(self, inputs, training = False):
        x = inputs

        for i in range(self.width):
            x = self.convs[i](x)

            x = self.activations[i](x, training = training)

        return x

    def get_config(self):
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "activation": self.activation,
            "width": self.width,
            "batch_norm": self.batch_norm,
            "dropout_rate": self.dropout_rate,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)