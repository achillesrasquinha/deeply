def DatasetGenerator(ds, mapper = None):
    if mapper:
        ds = ds.map(mapper)
    return ds