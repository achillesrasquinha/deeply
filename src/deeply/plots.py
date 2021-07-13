import matplotlib.pyplot as pplt

from deeply._compat import iteritems, iterkeys

def _matshow(axes, mat, title = None, axis = False, **kwargs):
    plot_args = kwargs.pop("plot_args", {})

    axes.matshow(mat, **plot_args)
    
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

def segplot(image, mask, predict = None, **kwargs):
    """
    Segmentation Plot
    """
    
    show_predict = predict is not None
    n_plots   = 3 if show_predict else 2

    fig, axes = pplt.subplots(1, n_plots, sharex = True, sharey = True)

    plot_args = { "cmap": pplt.cm.gray }

    _matshow(axes[0], mat = image, title = "Image")
    _matshow(axes[1], mat = mask,  title = "Mask", plot_args = plot_args)

    if show_predict:
        _matshow(axes[2], mat = predict, title = "Prediction", plot_args = plot_args)

    return _plot_base(fig, axes, **kwargs)

def history_plot(history, **kwargs):
    histories = history.history

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