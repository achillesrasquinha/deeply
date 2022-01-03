from tensorflow.keras.layers import (
    Layer,
    Activation,
    Dropout,
    BatchNormalization
)

class ActivationBatchNormDropout(Layer):
    """
        Activation -> Batch Normalization -> Dropout
    """
    def __init__(self, activation = "relu", batch_norm = True, dropout_rate = 0.5, *args, **kwargs):
        self._super = super(ActivationBatchNormDropout, self)
        self._super.__init__(*args, **kwargs)

        self.activation  = Activation(activation = activation) if activation else None
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