import collections

import numpy as np
import matplotlib.pyplot as pplt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from bpyutils._compat import iterkeys
from bpyutils.util.array import sequencify, squash

def _matshow(axes, mat, title = None, axis = False, **kwargs):
    plot_args = kwargs.pop("plot_args", {})

    print(mat.shape)

    axes.matshow(np.squeeze(mat), **plot_args)
    
    if title:
        axes.set_title(title)
        
    if not axis:
        axes.axis("off")

def _plot_base(fig, axes, super_title = None, to_file = None, figsize = None):
    if figsize:
        width, height = figsize
        fig.set_figwidth(width)
        fig.set_figheight(height)

    if super_title:
        fig.suptitle(super_title)

    if to_file:
        fig.savefig(to_file)

    pplt.tight_layout()

    return fig, axes

def _get_plot_kwargs(**kwargs):
    return {
        "cmap": kwargs.get("cmap", "gray")
    }

def segplot(image, mask, predict = None, **kwargs):
    """
    Segmentation Plot
    """
    
    show_predict = predict is not None
    n_plots   = 3 if show_predict else 2

    fig, axes = pplt.subplots(1, n_plots, sharex = True, sharey = True)

    plot_args = _get_plot_kwargs(**kwargs)

    _matshow(axes[0], mat = image, title = "Image")
    _matshow(axes[1], mat = mask,  title = "Mask", plot_args = plot_args)

    if show_predict:
        _matshow(axes[2], mat = predict, title = "Prediction", plot_args = plot_args)

    return _plot_base(fig, axes, **kwargs)

def imgplot(images, **kwargs):
    """
    Grid Plot
    """
    sequence = False

    if tf.is_tensor(images) or type(images) is np.ndarray:
        shape = images.shape

        if len(shape) > 3:
            length   = shape[0]
        else:
            length   = 1
            sequence = True

    if sequence or isinstance(images, (list, tuple)):
        images = sequencify(images)
        length = len(images)

    size = max(int(np.sqrt(length)), 1)

    fig, axes = pplt.subplots(size, size, sharex = True, sharey = True)
    plots = axes

    if not type(axes) is np.ndarray:
        plots = [sequencify(axes)]

    plot_args = _get_plot_kwargs(**kwargs)

    k = 0

    for i in range(size):
        for j in range(size):
            _matshow(plots[i][j], mat = images[k], plot_args = plot_args)
            k += 1

    return _plot_base(fig, squash(plots), **kwargs)

def history(obj, **kwargs):
    if isinstance(obj, collections.Mapping):
        histories = obj
    else:
        histories = obj.history

    metrics   = [m for m in iterkeys(histories) if not m.startswith("val_")]
    n_plots   = len(metrics)

    fig, axes = pplt.subplots(n_plots, 1, sharex = True)

    for i, type_ in enumerate(metrics):
        values  = histories[type_]
        axes[i].plot(values)
        
        val_key = "val_%s" % type_
        has_val = False

        if val_key in histories:
            values = histories[val_key]
            axes[i].plot(values)

            has_val = True

        axes[i].set_title(type_)

        legends = ["train"]
        if has_val:
            legends += ["validation"]

        axes[i].legend(legends)

    return _plot_base(fig, axes, **kwargs)

def confusion_matrix(y_true, y_pred):
    matrix = sk_confusion_matrix(y_true, y_pred)
    figure = sns.heatmap(matrix)
    return figure