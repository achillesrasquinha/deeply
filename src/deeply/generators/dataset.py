from tensorflow.data.experimental import (
    AUTOTUNE,
    cardinality
)

from deeply.const import DEFAULT

def DatasetGenerator(ds,
    cache      = True,
    shuffle    = True,
    batch_size = DEFAULT["batch_size"],
    mapper     = None
):
    if mapper:
        ds = ds.map(mapper, num_parallel_calls = AUTOTUNE)

    if cache or shuffle:
        ds = ds.cache()

        if shuffle:
            size = cardinality(ds).numpy()
            ds = ds.shuffle(size)

    if batch_size:
        ds = ds.batch(batch_size)

    ds = ds.prefetch(AUTOTUNE)

    return ds