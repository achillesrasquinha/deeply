
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint

from tqdm.keras import TqdmCallback

from deeply.callbacks.progress_step import ProgressStepCallback

from bpyutils.util.array import sequencify
from bpyutils.util.datetime import get_timestamp_str
from bpyutils._compat import iteritems

def get_checkpoint_prefix(model):
    prefix = "%s-%s" % (model.name or "model", get_timestamp_str(format_ = '%Y%m%d%H%M%S'))
    return prefix

def get_fit_kwargs(model, kwargs, custom = None):
    verbose = kwargs.pop("verbose", 0)
    monitor = kwargs.pop("monitor", "loss")

    callbacks = sequencify(kwargs.pop("callbacks", []))

    callbacks.append(ProgressStepCallback())
    # callbacks.append(TqdmCallback(verbose = verbose))

    callbacks.append(ModelCheckpoint(
        filepath            = "%s.hdf5" % get_checkpoint_prefix(model),
        monitor             = monitor,
        save_best_only      = True,
        save_weights_only   = True,
    ))

    kwargs["callbacks"] = callbacks

    if custom:
        for key, value in iteritems(custom):
            if key in ("callbacks",):
                value        = sequencify(value)
                kwarg_value  = sequencify(kwargs.pop(key, []))

                value       += kwarg_value
                
            kwargs[key] = value
            
    return kwargs

def get_input(x, y, channels):
    if not y:
        y = x

    input_shape = (x, y, channels)
    input_ = Input(shape = input_shape, name = "inputs")

    return input_