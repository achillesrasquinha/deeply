from deeply.const import DEFAULT

def DatasetGenerator(ds, batch_size = DEFAULT["batch_size"], mapper = None):
    if mapper:
        ds = ds.map(mapper)

    if batch_size:
        ds = ds.batch(batch_size)

    return ds