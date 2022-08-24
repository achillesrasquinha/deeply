from collections import Sequence

import numpy as np

from tensorflow.keras.layers import (
    Activation,
    Flatten,
    Reshape,
    Input,
    LeakyReLU
)

from deeply.const import DEFAULT
from deeply.model.base      import BaseModel
from deeply.model.layer     import DenseBlock, Conv2DTransposeBlock
from deeply.model.types     import is_layer_type
from deeply.util.model      import get_activation

from bpyutils.util._dict    import merge_dict
from bpyutils.util.imports  import import_handler

def GenerativeModel(
    x                   = None,
    y                   = None,
    channels            = 1,
    input_shape         = None,
    init_encoder_units  = 64,
    init_decoder_units  = 128,
    kernel_size         = 5,
    
    encoder_strides     = 2,
    decoder_strides     = 2,
    final_strides       = 2,

    padding             = "same",
    activation          = LeakyReLU,
    activation_args     = { "alpha": 0.2 },
    layer_width         = 1,
    layer_depth         = 2,
    output_resolution   = 0.25,
    encoder_layer_growth_rate = 1,
    decoder_layer_growth_rate = 1,
    feature_growth_rate = 1,
    minimum_features_x  = 5,
    minimum_features_y  = 5,
    encoder_fc_units    = 1,
    final_activation    = "sigmoid",
    final_units         = 1,
    latent_dim          = 100,

    encoder_dropout_rate = 0,
    encoder_batch_norm   = False,
    decoder_dropout_rate = 0,
    decoder_batch_norm   = False,

    kernel_initializer   = None,

    name                = "generative",
    encoder_name        = "encoder",
    decoder_name        = "decoder",

    backbone            = None,
    backbone_weights    = "imagenet",
    weights             = None,
    layer_block         = DenseBlock,

    model_type          = None,

    encoder_learning_rate = DEFAULT["generative_model_encoder_learning_rate"],
    decoder_learning_rate = DEFAULT["generative_model_decoder_learning_rate"],

    *args, **kwargs
):
    """
    Generative Adversarial Networks (GAN).

    :param x: Input features length.
    :param y: Input features height.
    :param channels: Number of channels for an input image.
    :param init_units: Number of neurons in the initial layer.
    :param activation: Activation function after each layer.
    :param activation_args: Arguments for the activation function.
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

    n_units = init_encoder_units

    activation = get_activation(activation, **(activation_args or {}))

    base_layer_args = dict(activation = activation, dropout_rate = encoder_dropout_rate,
        kernel_initializer = kernel_initializer, batch_norm = encoder_batch_norm,
        width = layer_width)

    layer_args = base_layer_args

    if is_layer_type(layer_block, "convolution"):
        layer_args = merge_dict(base_layer_args, {
            "kernel_size": kernel_size, "strides": encoder_strides, "padding": padding })

    if backbone:
        BackBone = import_handler("deeply.model.transfer.backbone.BackBone")
        backbone = BackBone(backbone, input_tensor = input_, input_shape = input_shape, weights = backbone_weights)
        input_   = backbone._model.input
        m        = backbone._model.output

        for _ in backbone.get_feature_layers():
            n_units = int(n_units * encoder_layer_growth_rate)

            if is_convolution:
                x = int(x * feature_growth_rate)
                y = int(y * feature_growth_rate)
    else:
        m = input_

        for _ in range(layer_depth):
            m = layer_block(n_units, **layer_args)(m)
            n_units = int(n_units * encoder_layer_growth_rate)

            if is_convolution:
                x = int(x * feature_growth_rate)
                y = int(y * feature_growth_rate)

    if is_convolution:
        x = max(minimum_features_x, x)
        y = max(minimum_features_y, y)

    final_block_args = merge_dict(base_layer_args, { "activation": final_activation })

    if is_convolution:
        m = Flatten()(m)
        m = DenseBlock(encoder_fc_units, **final_block_args)(m)

    # z_mean    = DenseBlock(latent_dim, **final_block_args, name = "z_mean")(m)
    # z_log_var = DenseBlock(latent_dim, **final_block_args, name = "z_log_var")(m)

    # z_mean, z_log_var = KLDivergence()([z_mean, z_log_var])
    
    # z         = Sampling(name = "z")([z_mean, z_log_var])

    encoder = BaseModel(inputs = [input_], outputs = m, name = "%s-%s" % (name, encoder_name))
    encoder.compile(learning_rate = encoder_learning_rate)

    decoder_input = Input(latent_dim)
    m = decoder_input

    # n_units = init_decoder_units // layer_growth_rate
    n_units = init_decoder_units

    layer_args["dropout_rate"] = decoder_dropout_rate
    layer_args["batch_norm"]   = decoder_batch_norm

    if is_convolution:
        x = int(x * output_resolution)
        y = int(y * output_resolution)

    if is_convolution:
        m = DenseBlock(x * y * n_units, **base_layer_args)(m)
        m = Reshape((x, y, n_units))(m)

        layer_block = Conv2DTransposeBlock

    if is_convolution:
        layer_args = merge_dict(layer_args, { "kernel_size": kernel_size, "width": 1,
            "use_bias": False })

    for i in range(layer_depth):
        if isinstance(decoder_strides, Sequence):
            layer_args["strides"] = decoder_strides[i]

        n_units = int(n_units * decoder_layer_growth_rate)
        m = layer_block(n_units, **layer_args)(m)

    if is_convolution:
        layer_args = merge_dict(layer_args, { "activation": None,
            "strides": final_strides, "kernel_size": (x, y) })

    m = layer_block(final_units, **layer_args)(m)
    output_layer = Activation(activation = final_activation, name = "output")(m)

    decoder = BaseModel(inputs = [decoder_input],
        outputs = [output_layer], name = "%s-%s" % (name, decoder_name))
    decoder.compile(learning_rate = decoder_learning_rate)

    model = model_type(encoder, decoder, name = name, **kwargs)

    if weights:
        model.load_weights(weights)

    model.compile()

    return model