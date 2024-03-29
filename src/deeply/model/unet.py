import os, os.path as osp
import json
import shutil

import numpy as np
from deeply.metrics import tversky_index
from deeply.model.transfer.backbone import BackBone

import tensorflow as tf
from tensorflow.data  import Dataset
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

from deeply.util.model      import get_checkpoint_prefix, get_input
from deeply.model.base      import BaseModel
from deeply.model.layer     import ActivationBatchNormDropout
from deeply.generators      import BaseDataGenerator
from deeply.callbacks       import GeneralizedEarlyStopping, PlotHistoryCallback
from deeply.metrics         import jaccard_index, dice_coefficient, tversky_index
from bpyutils.util.array    import sequencify
from bpyutils.util.system   import make_archive, make_temp_dir
from bpyutils.util.datetime import get_timestamp_str

# verify with paper...
def kernel_initializer(shape, dtype = None):
    n = np.prod(shape[:3])
    stddev = np.sqrt(2 / n)
    return tf.random.normal(shape, stddev = stddev, dtype = dtype)

class ConvBlock(Layer):
    def __init__(self, filters, kernel_size = 3, activation = "relu", width = 2, batch_norm = True,
        dropout_rate = 0.2, kernel_initializer = kernel_initializer, padding = "valid", *args, **kwargs):
        self._super = super(ConvBlock, self)
        self._super.__init__(*args, **kwargs)

        self.filters      = filters
        self.kernel_size  = kernel_size
        self.activation   = activation
        self.batch_norm   = batch_norm
        self.dropout_rate = dropout_rate
        self.padding      = padding
        self.kernel_initializer = kernel_initializer

        self.convs        = [ ]
        self.batch_norms  = [ ]
        self.activations  = [ ]
        self.dropouts     = [ ]

        for _ in range(width):
            conv = Conv2D(filters = filters, kernel_size = kernel_size,
                kernel_initializer = kernel_initializer, padding = padding)
            self.convs.append(conv)

            activation = ActivationBatchNormDropout(activation = activation,
                batch_norm = batch_norm, dropout_rate = dropout_rate)
            self.activations.append(activation)

        self.width = width

    # return x

    def call(self, inputs, training = False):
        x = inputs

        for i in range(self.width):
            x = self.convs[i](x, training = training)

            x = self.activations[i](x, training = training)

        return x

    def get_config(self):
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
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
        metrics.append(jaccard_index)
        metrics.append(tversky_index)

        kwargs["metrics"]   = metrics

        return self._super.compile(*args, **kwargs)

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
    attention_gate = None,
    backbone = None,
    backbone_weights = "imagenet",
    freeze_backbone  = False,
    weights = None,
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
    :param

    References
        [1]. Ronneberger, Olaf, et al. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” ArXiv:1505.04597 [Cs], May 2015. arXiv.org, http://arxiv.org/abs/1505.04597.
    
    >>> from deeply.model.unet import UNet
    >>> model = UNet()
    """
    input_shape = (x, y, channels)
    input_  = get_input(*input_shape)

    filters = init_filters
    conv_block_args = dict(kernel_size = kernel_size,
        activation = activation, dropout_rate = dropout_rate, width = n_conv,
        kernel_initializer = kernel_initializer, padding = padding, batch_norm = batch_norm)

    contracting_layers = [ ]

    if backbone:
        backbone = BackBone(backbone, input_tensor = input_, input_shape = input_shape, weights = backbone_weights)
        input_   = backbone._model.input
        m        = backbone._model.output

        for feature_layer in backbone.get_feature_layers():
            contracting_layers.append(feature_layer.output)
            filters = filters * filter_growth_rate
    else:
        m = input_

        # contracting path
        for _ in range(layer_depth):
            m = ConvBlock(filters = filters, **conv_block_args)(m)
            contracting_layers.append(m)
            m = MaxPooling2D(pool_size = pool_size, strides = mp_strides)(m)
            filters = filters * filter_growth_rate

    m = ConvBlock(filters = filters, **conv_block_args)(m)

    # expanding path
    for skip_layer in reversed(contracting_layers):
        filters = filters // filter_growth_rate
        m = Conv2DTranspose(filters = filters, kernel_size = up_conv_size,
            strides = pool_size, padding = padding,
            kernel_initializer = kernel_initializer)(m)
        m = Activation(activation = activation)(m)

        if attention_gate:
            skip_layer = attention_gate(skip_layer, m)

        m = copy_crop_concat_block(m, skip_layer)
        m = ConvBlock(filters = filters, **conv_block_args)(m)
    
    m = Conv2D(filters = n_classes, kernel_size = final_conv_size, padding = padding,
                kernel_initializer = kernel_initializer)(m)
    output_layer = Activation(activation = final_activation, name = "outputs")(m)

    model = UNetModel(inputs = [input_], outputs = [output_layer], name = name)

    if weights:
        model.load_weights(weights)

    return model

class AttentionGate(Layer):
    def __init__(self, batch_norm = True, dropout_rate = 0, *args, **kwargs):
        self.super       = super(AttentionGate, self)
        self.super.__init__(*args, **kwargs)

        self.batch_norm  = batch_norm
        self.droput_rate = dropout_rate

    def call(self, input_, gating_signal, training = False):
        t = Conv2D(filters = 1, kernel_size = (1, 1))(input_)
        g = Conv2D(filters = 1, kernel_size = (1, 1))(gating_signal)
        x = Add()([t, g])

        if training and self.batch_norm:
            x = BatchNormalization()(x)
        
        x = Activation("relu")(x)

        if training and self.droput_rate:
            x = Dropout(rate = self.droput_rate)(x)
        
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
    batch_norm   = kwargs.get("batch_norm", True)
    dropout_rate = kwargs.get("dropout_rate", 0)

    _attention_gate = kwargs.pop("attention_gate", AttentionGate(
        batch_norm = batch_norm, dropout_rate = dropout_rate))

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

def UNet3D(*args, **kwargs):
    """
    Constructs a 3D U-Net.
    
    >>> from deeply.model.unet import UNet3D
    >>> model = UNet3D()
    """
    unet = UNet(
        name = "unet-3d",
        **kwargs
    )

    return unet

def _center_crop(arr, shape):
    arr_shape = arr.shape

    diff_x = (arr_shape[0] - shape[0])
    diff_y = (arr_shape[1] - shape[1])

    assert diff_x >= 0
    assert diff_y >= 0

    if diff_x == 0 and diff_y == 0:
        return arr

    off_lx  = diff_x // 2
    off_ly  = diff_y // 2
    off_rx  = diff_x - off_lx
    off_ry  = diff_y - off_ly

    cropped = arr[ off_lx : -off_ly, off_rx : -off_ry ]

    return cropped

    # augmentor = iaa.Sequential([
    #     iaa.CenterCropToFixedSize(
    #         width  = shape[0],
    #         height = shape[1]
    #     )
    # ])

    # from_, to = (0, 1, 2), (1, 0, 2)

    # arr = arr.numpy()
    # arr = np.moveaxis(arr, from_, to)
    # aug = augmentor(images = [arr])
    # aug = squash(aug)

    # aug = np.moveaxis(aug, to, from_)

    # return aug

def _crop(shape):
    def crop(x, y):
        dtype = y.dtype
        label = tf.py_function(_center_crop, [y, shape], dtype)
        return x, label
    return crop

def _format_dataset(ds, mapper = None, target_shape = None, batch_size = 1, **kwargs):
    if isinstance(ds, Dataset):
        if mapper:
            ds = ds.map(mapper)

        if target_shape:
            ds = ds.map(_crop(target_shape))

        ds = ds.batch(batch_size)

    return ds

class Trainer:
    def __init__(self, artifacts_path = None):
        self.artifacts_path = osp.abspath(artifacts_path or get_timestamp_str('%Y%m%d%H%M%S'))

    def fit(self, model, train, val = None, batch_size = 32, early_stopping = True, monitor = "loss", **kwargs):
        target_shape = model.output_shape[1:]

        mapper = kwargs.pop("mapper", None)

        format_args = dict(target_shape = target_shape, mapper = mapper,
            batch_size = batch_size)

        train = _format_dataset(train, **format_args)
        if val:
            val = _format_dataset(val, **format_args)

        if isinstance(train, BaseDataGenerator):
            kwargs["steps_per_epoch"]  = train.n_samples // batch_size

        if isinstance(val, BaseDataGenerator):
            kwargs["validation_steps"] = val.n_samples // batch_size

        callbacks = sequencify(kwargs.get("callbacks", []))
        
        prefix    = get_checkpoint_prefix(model)

        if val:
            monitor = "val_%s" % monitor

        history = None

        with make_temp_dir() as tmp_dir:
            filepath   = osp.join(tmp_dir, "%s.hdf5" % prefix)
            checkpoint = ModelCheckpoint(
                filepath          = filepath,
                monitor           = monitor,
                save_best_only    = True,
                save_weights_only = True
            )

            callbacks.append(checkpoint)

            plothistory = PlotHistoryCallback(fpath = osp.join(tmp_dir, "history.png"))

            callbacks.append(plothistory)

            # if early_stopping:
            #     gen_early_stop = GeneralizedEarlyStopping(baseline = 0.05)
            #     callbacks.append(gen_early_stop)

            kwargs["callbacks"] = callbacks

            history  = model.fit(train, validation_data = val, **kwargs)

            filepath = osp.join(tmp_dir, "%s.json" % prefix)

            with open(filepath, mode = "w") as f:
                json.dump(history.history, f)
                
            make_archive(self.artifacts_path, "zip", tmp_dir)

        return history

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