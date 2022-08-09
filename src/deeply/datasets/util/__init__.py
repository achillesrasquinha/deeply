import numpy as np

from tensorflow.data.experimental import cardinality

from bpyutils.util.array import sequencify, squash

SPLIT_TYPES = ("train", "val", "test")

def length(ds):
    length = cardinality(ds)
    return length.numpy()

def split(ds, splits = (.6, .2, .2)):
    """
    Split a TensorFlow Dataset into splits.
    """
    if len(splits) == 1:
        splits = list(splits)
        splits = tuple(list(splits) + [1 - splits[0]])
    
    assert sum(splits) == 1
    
    ds_size = length(ds)
    curr_ds = ds

    for split in splits:
        size = np.round(split * ds_size)
        split_ds = curr_ds.take(size)
        
        yield split_ds
        
        curr_ds  = curr_ds.skip(size)

def concat(datasets, mapper = None):
    datasets = iter(sequencify(datasets))
    curr_ds  = next(datasets)

    if mapper:
        curr_ds = curr_ds.map(mapper)

    for dataset in datasets:
        if mapper:
            dataset = dataset.map(mapper)

        curr_ds = curr_ds.concatenate(dataset)

    return curr_ds