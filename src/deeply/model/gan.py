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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from deeply.util.model      import get_input
from deeply.model.base      import BaseModel
from deeply.model.layer     import ConvBlock, DenseBlock
from deeply.model.types     import is_layer_type
from deeply.model.autoencoder import AutoEncoder
from deeply.callbacks import GANPlotCallback
from deeply.const import DEFAULT

from bpyutils.util._dict    import merge_dict
from bpyutils.util.imports  import import_handler
from bpyutils.util.array    import sequencify

_binary_cross_entropy_logits = BinaryCrossentropy(from_logits = False)

def generator_loss(fake_output):
    return _binary_cross_entropy_logits(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = _binary_cross_entropy_logits(tf.ones_like(real_output), real_output)
    fake_loss = _binary_cross_entropy_logits(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss

class GANModel(AutoEncoder):
    def __init__(self, encoder, decoder, *args, **kwargs):
        self._super = super(GANModel, self)
        self._super.__init__(encoder, decoder, *args, **kwargs)

    @property
    def generator(self):
        return self.decoder

    @property
    def discriminator(self):
        return self.encoder

    def compile(self, *args, **kwargs):
        kwargs["loss"] = kwargs.get("loss", _binary_cross_entropy_logits)

        metrics = sequencify(kwargs.get("metrics", []))
        metrics.append(_binary_cross_entropy_logits)

        kwargs["metrics"] = metrics

        learning_rate     = kwargs.pop("learning_rate", DEFAULT["gan_learning_rate"])

        self.generator_optimizer     = Adam(learning_rate)
        self.discriminator_optimizer = Adam(learning_rate)

        self._super.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        callbacks = kwargs.pop("callbacks", [])
        callbacks.append(GANPlotCallback(self))

        kwargs["callbacks"] = callbacks

        return self._super.fit(*args, **kwargs) 

    def train_step(self, data):
        generator_input_shape = self.generator.input.shape
        _, noise_dim = generator_input_shape

        noise = tf.random.normal([1, noise_dim])

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated_output = self.generator(noise, training = True)

            real_output = self.discriminator(data, training = True)
            fake_output = self.discriminator(generated_output)

            loss_generator      = generator_loss(fake_output)
            loss_discriminator  = discriminator_loss(real_output, fake_output)

        generator_train_vars        = self.generator.trainable_variables
        discriminator_train_vars    = self.discriminator.trainable_variables

        generator_gradients     = generator_tape.gradient(loss_generator, generator_train_vars)
        discriminator_gradients = discriminator_tape.gradient(loss_discriminator, discriminator_train_vars)
        
        self.generator_optimizer.apply_gradients(zip(generator_gradients, generator_train_vars))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_train_vars))

        # self.compiled_metrics.update_state(data, y_pred)

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
    init_units          = 64,
    kernel_size         = 3,
    strides             = 2,
    padding             = "same",
    activation          = "relu",
    activation_args     = { },
    layer_width         = 1,
    layer_depth         = 2,
    output_resolution   = 0.25,
    layer_growth_rate   = 1,
    feature_growth_rate = 1,
    minimum_features_x  = 5,
    minimum_features_y  = 5,
    discriminator_fc_units = 1,
    final_activation    = "sigmoid",
    final_units         = 1,
    latent_dim          = 100,

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
    :param generator_fc_units: Number of neurons in fully-connected layer of the encoder.
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
        kernel_initializer = kernel_initializer, batch_norm = batch_norm, width = layer_width,
        activation_args = activation_args)

    layer_args = base_layer_args

    if is_layer_type(layer_block, "convolution"):
        layer_args = merge_dict(base_layer_args, {
            "kernel_size": kernel_size, "strides": strides, "padding": padding })

    if backbone:
        BackBone = import_handler("deeply.model.transfer.backbone.BackBone")
        backbone = BackBone(backbone, input_tensor = input_, input_shape = input_shape, weights = backbone_weights)
        input_   = backbone._model.input
        m        = backbone._model.output

        # for _ in backbone.get_feature_layers():
        #     n_units = int(n_units * layer_growth_rate)

            # if is_convolution:
            #     x = int(x * feature_growth_rate)
            #     y = int(y * feature_growth_rate)
    else:
        m = input_

        for _ in range(layer_depth):
            m = layer_block(n_units, **layer_args)(m)
            # n_units = int(n_units * layer_growth_rate)

            # if is_convolution:
            #     x = int(x * feature_growth_rate)
            #     y = int(y * feature_growth_rate)

    # if is_convolution:
    #     x = max(minimum_features_x, x)
    #     y = max(minimum_features_y, y)

    final_block_args = merge_dict(base_layer_args, { "activation": final_activation })

    if is_convolution:
        m = Flatten()(m)
        m = DenseBlock(discriminator_fc_units, **final_block_args)(m)

    # z_mean    = DenseBlock(latent_dim, **final_block_args, name = "z_mean")(m)
    # z_log_var = DenseBlock(latent_dim, **final_block_args, name = "z_log_var")(m)

    # z_mean, z_log_var = KLDivergence()([z_mean, z_log_var])
    
    # z         = Sampling(name = "z")([z_mean, z_log_var])

    discriminator = BaseModel(inputs = [input_], outputs = m, name = "%s-discriminator" % name)
    discriminator.compile()

    generator_input = Input(latent_dim)
    m = generator_input

    # n_units = n_units // layer_growth_rate
    n_units *= 2

    if is_convolution:
        x = int(x * output_resolution)
        y = int(y * output_resolution)

    if is_convolution:
        m = DenseBlock(x * y * n_units, **base_layer_args)(m)
        m = Reshape((x, y, n_units))(m)

        layer_block = Conv2DTranspose
        
        for key in ("width", "dropout_rate", "batch_norm", "activation_args"):
            layer_args.pop(key)

    if is_convolution:
        layer_args = merge_dict(layer_args, { "kernel_size": strides * 2 })

    for _ in range(layer_depth):
        m = layer_block(n_units, **layer_args)(m)
    #     n_units  = n_units // layer_growth_rate

    if is_convolution:
        layer_args = merge_dict(layer_args, { "activation": None,
            "strides": 1, "kernel_size": (x, y) })

    m = layer_block(final_units, **layer_args)(m)
    output_layer = Activation(activation = final_activation, name = "outputs")(m)

    generator = BaseModel(inputs = [generator_input], outputs = [output_layer], name = "%s-generator" % name)

    model = GANModel(discriminator, generator, name = name)

    if weights:
        model.load_weights(weights)

    model.compile()

    return model

def DCGAN(**kwargs):
    """
    Constructs a Deep Convolutional Generative Adversarial Network.

    References
        [1]. https://arxiv.org/abs/1511.06434 
        
    >>> import deeply
    >>> model = deeply.hub("convolutional-generative-adversarial-network")
    """
    model = GAN(
        name = "dcgan",
        layer_block = ConvBlock,
        **kwargs
    )

    return model