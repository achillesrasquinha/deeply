import numpy as np

from tensorflow.data.experimental import cardinality

def split(ds, splits = (.6, .2, .2)):
    """
    Split a TensorFlow Dataset into splits.
    """
    assert sum(splits) == 1
    
    ds_size = cardinality(ds).numpy()
    curr_ds = ds

    for split in splits:
        size = np.round(split * ds_size)
        split_ds = curr_ds.take(size)
        
        yield split_ds
        
        curr_ds  = curr_ds.skip(size)