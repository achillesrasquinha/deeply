import numpy as np

import tensorflow as tf
from tensorflow.data  import Dataset
from tensorflow.keras import Input
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import (
    Layer,
    Conv2D,
    MaxPooling2D,
    Activation,
    Cropping2D,
    Conv2DTranspose,
    Dropout,
    Concatenate,
    Add,
    Multiply,
    BatchNormalization
)
from tensorflow.keras.callbacks import ModelCheckpoint

import imgaug.augmenters as iaa

from deeply.model.base      import BaseModel
from deeply.metrics         import dice_coefficient
from deeply.util.array      import sequencify, squash
from deeply.util.datetime   import get_timestamp_str

def kernel_initializer(shape, dtype = None):
    n = np.prod(shape[:3])
    stddev = np.sqrt(2 / n)
    return tf.random.normal(shape, stddev = stddev, dtype = dtype)

class ConvBlock(Layer):
    def __init__(self, filters, kernel_size = 3, activation = "relu", width = 2, batch_norm = True,
        dropout_rate = 0.2, kernel_initializer = kernel_initializer, padding = "valid", *args, **kwargs):
        self._super = super(ConvBlock, self)
        self._super.__init__(*args, **kwargs)

        self.dropout_rate = dropout_rate

        self.convs        = [ ]
        self.batch_norms  = [ ]
        self.activations  = [ ]
        self.dropouts     = [ ]

        for _ in range(width):
            conv = Conv2D(filters = filters, kernel_size = kernel_size,
                kernel_initializer = kernel_initializer, padding = padding)
            self.convs.append(conv)

            # https://stackoverflow.com/a/40295999
            if batch_norm:
                bn = BatchNormalization()
                self.batch_norms.append(bn)

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

            if training and self.batch_norms:
                x = self.batch_norms[i](x)

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

        metrics             = sequencify(kwargs.get("metrics", []))
        if kwargs["loss"] == "categorical_crossentropy" and not metrics:
            metrics.append("categorical_accuracy")
            
        metrics.append(dice_coefficient)

        kwargs["metrics"]   = metrics

        callbacks           = sequencify(kwargs.get("callbacks", []))
        
        filepath   = "%s-%s.hdf5" % (self.name or "model", get_timestamp_str())
        checkpoint = ModelCheckpoint(
            filepath          = filepath,
            monitor           = "loss",
            save_best_only    = True,
            save_weights_only = True
        )

        callbacks.append(checkpoint)

        kwargs["callbacks"] = callbacks

        return self._super.compile(*args, **kwargs)

    # def fit(self, *args, **kwargs):
    #     return self._super.fit(*args, **kwargs)

    def plot(self, *args, **kwargs):
        kwargs["show_shapes"] = kwargs.get("show_shapes", True)
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
    batch_norm   = True, # recommendation, don't use batch norm and dropout at the same time.
    dropout_rate = 0,
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
    :param batch_norm: Batch Normalization after each convolution.
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
        kernel_initializer = kernel_initializer, padding = padding, batch_norm = batch_norm)

    contracting_layers = [ ]

    # contracting path
    for _ in range(layer_depth):
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

    if batch_norm:
        m = BatchNormalization()(m)

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
    _attention_gate = kwargs.pop("attention_gate", attention_gate)

    unet = UNet(
        name = "attention-unet",
        attention_gate = _attention_gate,
        **kwargs
    )

    return unet

def UnetPP(*args, **kwargs):
    """
    Constructs a U-Net++.

    >>> from deeply.model.unet import UnetPP
    >>> model = UnetPP()
    """
    layer_depth = kwargs.get("layer_depth", 4)
    unets = [ ]

    for i in range(layer_depth):
        unet = UNet(
            name = "unet++-%s" % i,
            layer_depth = i,
            **kwargs
        )
        unets.append(unet)

    unetpp = None

    return unetpp

def _center_crop(arr, shape):
    # arr_shape = arr.shape

    # diff_x = (arr_shape[0] - shape[0])
    # diff_y = (arr_shape[1] - shape[1])

    # assert diff_x >= 0
    # assert diff_y >= 0

    # if diff_x == 0 and diff_y == 0:
    #     return arr

    # off_lx  = diff_x // 2
    # off_ly  = diff_y // 2
    # off_rx  = diff_x - off_lx
    # off_ry  = diff_y - off_ly

    # cropped = arr[ off_lx : -off_ly, off_rx : -off_ry ]

    # return cropped

    augmentor = iaa.Sequential([
        iaa.CenterCropToFixedSize(
            width  = shape[0],
            height = shape[1]
        )
    ])

    from_, to = (0, 1, 2), (1, 0, 2)

    arr = arr.numpy()
    arr = np.moveaxis(arr, from_, to)
    aug = augmentor(images = [arr])
    aug = squash(aug)

    aug = np.moveaxis(aug, to, from_)

    return aug

def _crop(shape):
    def crop(x, y):
        dtype = y.dtype
        label = tf.py_function(_center_crop, [y, shape], dtype)
        return x, label
    return crop

def _format_dataset(ds, mapper = None, target_shape = None, batch_size = 1, **kwargs):
    if mapper:
        ds = ds.map(mapper)

    if target_shape:
        ds = ds.map(_crop(target_shape))

    return ds.batch(batch_size)
class Trainer:
    def fit(self, model, train, val = None, batch_size = 32, **kwargs):
        target_shape = model.output_shape[1:]

        mapper = kwargs.pop("mapper", None)

        format_args = dict(target_shape = target_shape, mapper = mapper,
            batch_size = batch_size)

        train = _format_dataset(train, **format_args)
        if val:
            val = _format_dataset(val, **format_args)

        return model.fit(train, validation_data = val, **kwargs)

def _generate_samples(x = 200, y = None, channels = 1, n_samples = 100,
    r_min_f = 1, r_max_f = 10, seg_min_f = 1, seg_max_f = 5):
    if not y:
        y = x

    features, labels = np.empty((n_samples, y, x, channels)), \
        np.empty((n_samples, y, x))
    
    size = min(x, y)
    compute_factor = lambda f: (f / 100) * size
    min_radius = compute_factor(r_min_f)
    max_radius = compute_factor(r_max_f)
    
    min_segs   = compute_factor(seg_min_f)
    max_segs   = compute_factor(seg_max_f)

    for i in range(n_samples):
        feature = np.ones((y, x, channels))
        mask    = np.zeros((y, x), dtype = np.bool)

        n_segs  = np.random.randint(min_segs, max_segs)

        for _ in range(n_segs):
            from_       = np.random.randint(0, x)
            to          = np.random.randint(0, y)

            radius      = np.random.randint(min_radius, max_radius)

            cx, cy      = np.ogrid[-to:y-to, -from_:x-from_]
            circle      = cx*cx + cy*cy <= radius*radius

            color       = np.random.randint(1, 255)
            
            mask        = np.logical_or(mask, circle)
            
            feature[circle] = color

        # noise
        feature += np.random.normal(size = feature.shape)
        feature -= np.amin(feature)
        feature /= np.amax(feature)

        features[i] = feature
        labels[i]   = mask

    return features, labels

def generate_toy(splits = (.6, .2, .2), n_samples = 100, **kwargs):
    """
    Create a toy dataset for U-Net Image Segmentation.

    :param x: Width of image.
    :param y: Height of image.
    :param channels: Number of channels in image.
    :param n_samples: Number of samples to generate.
    :param splits: Number of dataset splits.
    """
    return [
        Dataset.from_tensor_slices(_generate_samples(n_samples = int(split * n_samples), **kwargs))
            for split in splits
    ]