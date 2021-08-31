import numpy as np

from tensorflow.data.experimental import cardinality

from deeply.util.array import sequencify

def split(ds, splits = (.6, .2, .2)):
    """
    Split a TensorFlow Dataset into splits.
    """
    if len(splits) == 1:
        splits = list(splits)
        splits = tuple(list(splits) + [1 - splits[0]])
    
    assert sum(splits) == 1
    
    ds_size = cardinality(ds).numpy()
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