from tensorflow.keras.callbacks import Callback

from deeply.plots   import history as history_plot
from bpyutils._compat import iteritems

class PlotHistoryCallback(Callback):
    def __init__(self, fpath = "history.png", *args, **kwargs):
        self._super = super(PlotHistoryCallback, self)
        self._super.__init__(*args, **kwargs)

        self.fpath  = fpath
        self.logs   = { }

    def on_epoch_end(self, epoch, logs = None):
        if logs:
            for type_, value in iteritems(logs):
                vals = self.logs.get(type_, [ ])
                vals.append(value)

                self.logs[type_] = vals

            history_plot(self.logs, to_file = self.fpath, figsize = (8, 32))