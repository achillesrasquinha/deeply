import numpy as np

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import (
    Layer,
    Conv2D,
    MaxPooling2D,
    Activation,
    Cropping2D,
    Conv2DTranspose,
    UpSampling2D,
    Dropout,
    Concatenate,
    Add,
    Multiply,
)

from deeply.model.base import BaseModel

def kernel_initializer(shape, dtype = None):
    n = np.prod(shape[:3])
    stddev = np.sqrt(2 / n)
    return tf.random.normal(shape, stddev = stddev, dtype = dtype)

class ConvBlock(Layer):
    def __init__(self, filters, kernel_size = 3, activation = "relu", width = 2, 
        dropout_rate = 0.2, kernel_initializer = kernel_initializer, padding = "valid", *args, **kwargs):
        self._super = super(ConvBlock, self)
        self._super.__init__(*args, **kwargs)

        self.dropout_rate = dropout_rate

        self.convs        = [ ]
        self.activations  = [ ]
        self.dropouts     = [ ]

        for _ in range(width):
            conv = Conv2D(filters = filters, kernel_size = kernel_size,
                kernel_initializer = kernel_initializer, padding = padding)
            self.convs.append(conv)

            activation = Activation(activation = activation)
            self.activations.append(activation)

            # https://stats.stackexchange.com/a/317313
            if dropout_rate:
                dropout = Dropout(rate = dropout_rate)
                self.dropouts.append(dropout)

        self.width = width

    # return x

    def call(self, inputs, training = False):
        x = inputs

        for i in range(self.width):
            x = self.convs[i](x)
            x = self.activations[i](x)

            if training and self.dropouts:
                x = self.dropouts[i](x)

        return x

def get_crop_length(a, b):
    c = a - b
    assert (c >= 0)
    if c % 2 != 0:
        c1, c2 = int(c/2), int(c/2) + 1
    else:
        c1, c2 = int(c/2), int(c/2)

    return (c1, c2)

def get_crop_shape(a, b):
    a_shape = int_shape(a)
    b_shape = int_shape(b)

    cw1, cw2 = get_crop_length(a_shape[2], b_shape[2])
    ch1, ch2 = get_crop_length(a_shape[1], b_shape[1])

    return (ch1, ch2), (cw1, cw2)

def copy_crop_concat_block(x, skip_layer, **kwargs):
    ch, cw = get_crop_shape(skip_layer, x)
    skip_layer_cropped = Cropping2D(cropping = (ch, cw))(skip_layer)

    x = Concatenate()([x, skip_layer_cropped])

    return x

class UNetModel(BaseModel):
    def __init__(self, *args, **kwargs):
        self._super = super(UNetModel, self)
        self._super.__init__(*args, **kwargs)

    def compile(self, *args, **kwargs):
        kwargs["optimizer"] = kwargs.get("optimizer", "sgd")
        kwargs["loss"]      = kwargs.get("loss", "categorical_crossentropy")

        return self._super.compile(*args, **kwargs)

    def plot(self, *args, **kwargs):
        return self._super.plot(*args, **kwargs)

def UNet(
    x = None,
    y = None,
    channels     = 1,
    n_classes    = 2,
    layer_depth  = 4,
    n_conv       = 2,
    kernel_size  = 3,
    init_filters = 64,
    filter_growth_rate = 2,
    activation   = "relu",
    padding      = "valid",
    dropout_rate = 0.2,
    pool_size    = 2,
    mp_strides   = 2,
    up_conv_size = 2,
    final_conv_size  = 1,
    final_activation = "softmax",
    kernel_initializer = kernel_initializer,
    name = "unet",
    attention_gate = None
):
    """
    Constructs a U-Net.

    :param x: Input image width.
    :param y: Input image height.
    :param channels: Number of channels for input image.

    :param layer_depth: Depth of the U-Net.
    :param n_conv: Number of convolutions in each layer.
    :param kernel_size: Size of kernel in a convolution.
    :param init_filters: Number of filters in initial convolution.
    :param filter_growth_rate: Growth rate of filter over convolutions.
    :param activation: Activation function after each convolution.
    :param dropout_rate: Dropout rate after each convolution.
    :param pool_size: Size of max pooling layer.
    :param mp_strides: Size of strides of max pooling layer.
    :param up_conv_size: Size of upsampling layer.
    :param final_activation: Activation function on final layer.
    :param final_conv_size: Kernel size of final convolution.
    :param kernel_initializer: Weight initializer for each convolution block.
    :param attention_gate: Use a custom attention gate.

    References
        [1]. Ronneberger, Olaf, et al. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” ArXiv:1505.04597 [Cs], May 2015. arXiv.org, http://arxiv.org/abs/1505.04597.
    
    >>> from deeply.model.unet import UNet
    >>> model = UNet()
    """
    if not y:
        x = y

    input_shape = (x, y, channels)
    input_ = Input(shape = input_shape, name = "inputs")

    m = input_

    filters = init_filters
    conv_block_args = dict(kernel_size = kernel_size,
        activation = activation, dropout_rate = dropout_rate, width = n_conv,
        kernel_initializer = kernel_initializer, padding = padding)

    contracting_layers = [ ]

    # contracting path
    for depth in range(layer_depth):
        m = ConvBlock(filters = filters, **conv_block_args)(m)
        contracting_layers.append(m)
        filters = filters * filter_growth_rate
        m = MaxPooling2D(pool_size = pool_size, strides = mp_strides)(m)

    m = ConvBlock(filters = filters, **conv_block_args)(m)

    # expanding path
    for skip_layer in reversed(contracting_layers):
        filters = filters // filter_growth_rate
        m = Conv2DTranspose(filters = filters, kernel_size = up_conv_size,
            strides = pool_size, padding = padding,
            kernel_initializer = kernel_initializer)(m)

        if attention_gate:
            skip_layer = attention_gate(input_ = skip_layer, gating_signal = m)

        m = copy_crop_concat_block(m, skip_layer)
        m = ConvBlock(filters = filters, **conv_block_args)(m)
    
    m = Conv2D(filters = n_classes, kernel_size = final_conv_size, padding = padding,
                kernel_initializer = kernel_initializer)(m)
    m = Activation(activation = activation)(m)

    output_layer = Activation(activation = final_activation, name = "outputs")(m)

    model = UNetModel(inputs = input_, outputs = output_layer, name = name)

    return model

def attention_gate(input_, gating_signal):
    t = Conv2D(filters = 1, kernel_size = (1, 1))(input_)
    g = Conv2D(filters = 1, kernel_size = (1, 1))(gating_signal)
    x = Add()([t, g])
    
    x = Activation("relu")(x)
    
    x = Conv2D(filters = 1, kernel_size = (1, 1))(x)
    x = Activation("sigmoid")(x)
    
    x = Multiply()([input_, x])

    return x

def AttentionUNet(*args, **kwargs):
    """
    Constructs an Attention U-Net.

    References
        [1]. Oktay, Ozan, et al. “Attention U-Net: Learning Where to Look for the Pancreas.” ArXiv:1804.03999 [Cs], May 2018. arXiv.org, http://arxiv.org/abs/1804.03999.

    >>> from deeply.model.unet import AttentionUNet
    >>> model = AttentionUNet()
    """
    unet = UNet(
        name = "attention-unet",
        attention_gate = attention_gate
    )

    return unet

def generate_toy(x = 32, y = None, n_samples = 100,
    r_min_f = 1, r_max_f = 10, seg_min_f = 1, seg_max_f = 5):
    """
    Create a toy dataset for U-Net Image Segmentation.

    :param x: Width of image.
    :param y: Height of image.
    :param channels: Number of channels in image.
    :param n_samples: Number of samples to generate.
    """
    if not y:
        y = x

    channels = 1

    features, labels = np.zeros((n_samples, y, x, channels), dtype = np.uint8), \
        np.zeros((n_samples, y, x), dtype = np.bool)
    
    size = min(x, y)
    compute_factor = lambda f: (f / 100) * size
    min_radius = compute_factor(r_min_f)
    max_radius = compute_factor(r_max_f)
    
    min_segs   = compute_factor(seg_min_f)
    max_segs   = compute_factor(seg_max_f)

    for i in range(n_samples):
        feature = np.ones((y, x, channels))
        n_segs  = np.random.randint(min_segs, max_segs)

        mask    = np.zeros((y, x), dtype = np.bool)

        for _ in range(n_segs):
            from_       = np.random.randint(0, x)
            to          = np.random.randint(0, y)

            radius      = np.random.randint(min_radius, max_radius)

            cx, cy      = np.ogrid[-to:y-to, -from_:x-from_]
            circle      = cx*cx + cy*cy <= radius*radius

            color       = np.random.randint(1, 255)
            
            mask        = np.logical_or(mask, circle)
            
            feature[circle] = color

        features[i] = feature
        labels[i]   = mask

    return features, labels