from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.data import Dataset

from tqdm.keras import TqdmCallback

from deeply.callbacks.plots.history import PlotHistoryCallback
from deeply.callbacks.progress_step import ProgressStepCallback
from deeply.const import DEFAULT
from deeply.datasets.util import length as dataset_length

from bpyutils.util.array import sequencify
from bpyutils.util.datetime import get_timestamp_str
from bpyutils._compat import iteritems
from bpyutils.util._dict import merge_dict
from bpyutils.log import get_logger

import numpy as np
import pandas as pd

logger = get_logger()

def get_checkpoint_prefix(model):
    prefix = "%s-%s" % (model.name or "model", get_timestamp_str(format_ = '%Y%m%d%H%M%S'))
    return prefix

def _convert_ds(ds, mapper = None, batch_size = 1, **kwargs):
    if mapper:
        ds = ds.map(mapper)

    ds = ds.batch(batch_size)

    return ds

def get_fit_args_kwargs(model, args, kwargs, custom = None):
    mapper      = kwargs.pop("mapper", None)
    batch_size  = kwargs.pop("batch_size", DEFAULT["batch_size"])

    _convert_ds_kwargs = dict(mapper = mapper, batch_size = batch_size)

    if args:
        args = list(args)

        arg  = args[0]
        if isinstance(arg, Dataset):
            args[0] = _convert_ds(arg, **_convert_ds_kwargs)
            
            kwargs["steps_per_epoch"] = min(1, dataset_length(args[0]) // batch_size)
        
        args = tuple(args)

    verbose = kwargs.pop("verbose", 1)
    monitor = kwargs.pop("monitor", "val_loss")
    logger.info("Monitoring %s..." % monitor)
    use_multiprocessing = kwargs.pop("use_multiprocessing", True)

    callbacks = sequencify(kwargs.pop("callbacks", []))

    # callbacks.append(PlotHistoryCallback())
    # callbacks.append(ProgressStepCallback())
    callbacks.append(TqdmCallback()) # verbose = verbose

    early_stopping = kwargs.pop("early_stopping", None)
    if early_stopping:
        callbacks.append(EarlyStopping(**early_stopping))

    callbacks.append(ModelCheckpoint(
        filepath            = kwargs.pop("checkpoint_path", "%s.hdf5" % get_checkpoint_prefix(model)),
        monitor             = monitor,
        save_best_only      = True,
        save_weights_only   = True,
    ))

    kwargs["batch_size"] = batch_size
    kwargs["verbose"]    = 0
    kwargs["callbacks"]  = callbacks
    kwargs["use_multiprocessing"] = use_multiprocessing

    if custom:
        for key, value in iteritems(custom):
            if key in ("callbacks",):
                value        = sequencify(value)
                kwarg_value  = sequencify(kwargs.pop(key, []))

                value       += kwarg_value
                
            kwargs[key] = value
            
    return args, kwargs

def get_input(x, y, channels):
    if not y:
        y = x

    input_shape = (x, y, channels)
    input_ = Input(shape = input_shape, name = "inputs")

    return input_

def create_model_fn(func, doc = "", args = {}):
    def wrapper(**kwargs):
        kwargs   = merge_dict(args, kwargs)

        function = func(**kwargs)
        function.__doc__ = doc

        return function

    return wrapper

def get_activation(activation, **kwargs):
    if not isinstance(activation, str):
        return activation(**kwargs)

    return activation

def update_kwargs(kwargs, custom):
    for key, config in iteritems(custom):
        prev = kwargs.pop(key, config.get("default", None))

        item = config["item"]
        
        if isinstance(prev, list):
            item  = sequencify(item)
            prev += item
        else:
            prev  = item

        kwargs[key] = prev

    return kwargs