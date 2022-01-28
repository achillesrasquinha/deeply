import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Layer,
    Activation,
    Conv2DTranspose,
    Flatten,
    Reshape,
    Input
)
from tensorflow.keras.losses import binary_crossentropy

from deeply.util.model      import get_input
from deeply.model.base      import BaseModel
from deeply.model.layer     import ConvBlock, DenseBlock
from deeply.model.types     import is_layer_type
from deeply.model.autoencoder import AutoEncoder
from deeply.losses          import bernoulli_loss

from bpyutils.util._dict    import merge_dict
from bpyutils.util.imports  import import_handler
from bpyutils.util.array    import sequencify

class VAEModel(AutoEncoder):
    def __init__(self, encoder, decoder, *args, **kwargs):
        self._super = super(VAEModel, self)
        self._super.__init__(encoder, decoder, *args, **kwargs)

    def compile(self, *args, **kwargs):
        kwargs["loss"] = kwargs.get("loss", bernoulli_loss)

        metrics        = sequencify(kwargs.get("metrics", []))
        metrics.append(bernoulli_loss)

        kwargs["metrics"] = metrics

        self._super.compile(*args, **kwargs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            y_pred = self.decoder(z)

            loss   = bernoulli_loss(data, y_pred)

        trainable_weights   = self.trainable_weights
        gradients           = tape.gradient(loss, trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, trainable_weights))

        self.compiled_metrics.update_state(data, y_pred)

        return { m.name: m.result() for m in self.metrics }

class KLDivergence(Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True

        self._super = super(KLDivergence, self)
        self._super.__init__(*args, **kwargs)

    def call(self, inputs):
        mean, log_var = inputs

        kl_divergence = -.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis = -1)

        self.add_loss(K.mean(kl_divergence), inputs = inputs)

        return inputs

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_sigma = K.exp(0.5 * z_log_var)

        shape   = tf.shape(z_mean)

        epsilon = K.random_normal(shape = shape)

        return z_mean + z_sigma * epsilon

def GAN(
    x                   = None,
    y                   = None,
    channels            = 1,
    input_shape         = None,
    init_units          = 32,
    kernel_size         = 3,
    strides             = 2,
    padding             = "same",
    activation          = "relu",
    layer_width         = 1,
    layer_depth         = 2,
    layer_growth_rate   = 2,
    feature_growth_rate = 0.5,
    minimum_features_x  = 5,
    minimum_features_y  = 5,
    encoder_fc_units    = 16,
    final_activation    = "sigmoid",
    final_units         = 1,
    latent_dim          = 2,

    batch_norm          = True,
    dropout_rate        = 0,

    kernel_initializer  = None,

    name                = "gan",

    backbone            = None,
    backbone_weights    = "imagenet",
    weights             = None,
    layer_block         = DenseBlock,

    *args, **kwargs
):
    """
    Generative Adversarial Networks (GAN).

    :param x: Input features length.
    :param y: Input features height.
    :param channels: Number of channels for an input image.
    :param init_units: Number of neurons in the initial layer.
    :param activation: Activation function after each layer.
    :param kernel_initializer: Weight initializer for each layer.
    :param layer_width: Width of each layer.
    :param layer_growth_rate: Growth rate of layers.
    :param feature_growth_rate: Growth rate of input features.
    :param minimum_features: Minimum input features.
    :param layer_depth: Depth of the VAE.
    :param latent_dim: Latent Dimensions of the VAE.

    :param kernel_size: Size of kernel in a convolution.
    :param strides: Size of strides in a convolution.
    :param padding: Padding type in a convolution.
    :param encoder_fc_units: Number of neurons in fully-connected layer of the encoder.
    :param final_units: Number of neurons in final layer.

    :param dropout_rate: Dropout rate after each layer.
    :param batch_norm: Batch Normalization after each layer.

    :param n_dense: Number of dense sub-layers in each layer.

    :param pool_size: Size of max pooling layer.
    :param mp_strides: Size of strides of max pooling layer.
    :param up_conv_size: Size of upsampling layer.
    :param final_activation: Activation function on final layer.
    :param final_conv_size: Kernel size of final convolution.
    :param attention_gate: Use a custom attention gate.

    References
        [1]. Kingma, Diederik P., and Max Welling. “Auto-Encoding Variational Bayes.” ArXiv:1312.6114 [Cs, Stat], May 2014. arXiv.org, http://arxiv.org/abs/1312.6114.

    >>> from deeply.model.vae import VAE
    >>> model = VAE()
    """
    is_convolution = is_layer_type(layer_block, "convolution")

    if not input_shape:
        if is_convolution:
            if not y:
                y = x

            input_shape = (x, y, channels)
        else:
            shape = []

            for dim in (x, y):
                if dim:
                    shape.append(dim)

            dim = np.prod(shape)

            input_shape = (dim,)

    input_  = Input(shape = input_shape)

    n_units = init_units

    base_layer_args = dict(activation = activation, dropout_rate = dropout_rate,
        kernel_initializer = kernel_initializer, batch_norm = batch_norm, width = layer_width)

    layer_args = base_layer_args

    if is_layer_type(layer_block, "convolution"):
        layer_args = merge_dict(base_layer_args, {
            "kernel_size": kernel_size, "strides": strides, "padding": padding })

    if backbone:
        BackBone = import_handler("deeply.model.transfer.backbone.BackBone")
        backbone = BackBone(backbone, input_tensor = input_, input_shape = input_shape, weights = backbone_weights)
        input_   = backbone._model.input
        m        = backbone._model.output

        for _ in backbone.get_feature_layers():
            n_units = int(n_units * layer_growth_rate)

            if is_convolution:
                x = int(x * feature_growth_rate)
                y = int(y * feature_growth_rate)
    else:
        m = input_

        for _ in range(layer_depth):
            m = layer_block(n_units, **layer_args)(m)
            n_units = int(n_units * layer_growth_rate)

            if is_convolution:
                x = int(x * feature_growth_rate)
                y = int(y * feature_growth_rate)

    if is_convolution:
        x = max(minimum_features_x, x)
        y = max(minimum_features_y, y)

    final_block_args = merge_dict(base_layer_args, { "activation": None })

    if is_convolution:
        m = Flatten()(m)
        m = DenseBlock(encoder_fc_units, **final_block_args)(m)

    z_mean    = DenseBlock(latent_dim, **final_block_args, name = "z_mean")(m)
    z_log_var = DenseBlock(latent_dim, **final_block_args, name = "z_log_var")(m)

    z_mean, z_log_var = KLDivergence()([z_mean, z_log_var])
    
    z         = Sampling(name = "z")([z_mean, z_log_var])

    encoder   = BaseModel(inputs = [input_], outputs = [z_mean, z_log_var, z], name = "%s-encoder" % name)

    decoder_input = Input(shape = (latent_dim,))
    m         = decoder_input

    n_units = n_units // layer_growth_rate

    if is_convolution:
        m = DenseBlock(x * y * n_units, **base_layer_args)(m)
        m = Reshape((x, y, n_units))(m)

        layer_block = Conv2DTranspose
        
        for key in ("width", "dropout_rate", "batch_norm"):
            layer_args.pop(key)

    for _ in range(layer_depth):
        m = layer_block(n_units, **layer_args)(m)
        n_units  = n_units // layer_growth_rate

    if is_convolution:
        layer_args = merge_dict(layer_args, { "activation": None,
            "strides": 1 })

    m = layer_block(final_units, **layer_args)(m)
    output_layer = Activation(activation = final_activation, name = "outputs")(m)

    decoder = BaseModel(inputs = [decoder_input], outputs = [output_layer], name = "%s-decoder" % name)

    model   = VAEModel(encoder, decoder, name = name)

    if weights:
        model.load_weights(weights)

    model.compile()

    return model

def DCGAN(*args, **kwargs):
    """
    Constructs a Convolutional VAE.

    References
        [1]. Kingma, Diederik P., and Max Welling. “Auto-Encoding Variational Bayes.” ArXiv:1312.6114 [Cs, Stat], May 2014. arXiv.org, http://arxiv.org/abs/1312.6114.

    >>> import deeply
    >>> model = deeply.hub("convolutional-variational-autoencoder")
    """
    model = GAN(
        name = "dcgan",
        layer_block = ConvBlock,
        **kwargs
    )

    return model