import matplotlib.pyplot as pplt

from deeply._compat import iteritems, iterkeys

def _matshow(axes, mat, title = None, axis = False, **kwargs):
    plot_args = kwargs.pop("plot_args", {})

    axes.matshow(mat, **plot_args)
    
    if title:
        axes.set_title(title)
        
    if not axis:
        axes.axis("off")

def segplot(image, mask, predict = None):
    show_predict = predict is not None
    n_plots   = 3 if show_predict else 2

    _, axes   = pplt.subplots(1, n_plots, sharex = True, sharey = True)

    plot_args = { "cmap": pplt.cm.gray }

    _matshow(axes[0], mat = image, title = "Image")
    _matshow(axes[1], mat = mask,  title = "Mask", plot_args = plot_args)

    if show_predict:
        _matshow(axes[2], mat = predict, title = "Prediction", plot_args = plot_args)

    pplt.tight_layout()

def history_plot(history):
    histories = history.history

    metrics   = [m for m in iterkeys(histories) if not m.startswith("val_")]
    n_plots   = len(metrics)

    _, axes   = pplt.subplots(n_plots, 1, sharex = True)

    for i, type_ in enumerate(metrics):
        values  = histories[type_]
        axes[i].plot(values)
        
        val_key = "val_%s" % type_
        has_val = False

        if val_key in histories:
            values = histories[type_]
            axes[i].plot(values)

            has_val = True

        axes[i].set_title(type_)

        legends = ["train"]
        if has_val:
            legends += ["validation"]

        axes[i].legend(legends)

    pplt.tight_layout()