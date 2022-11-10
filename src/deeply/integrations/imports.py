from bpyutils.util.imports import import_handler
from bpyutils.log import get_logger

from deeply.__attr__ import __name__ as NAME

logger = get_logger(NAME)

_DASK_CLIENT = None

def _get_dask_client():
    global _DASK_CLIENT

    if _DASK_CLIENT is None:
        from dask.distributed import Client

        client = Client()
        logger.info("Created Dask client: %s" % client)

        _DASK_CLIENT = client

    return _DASK_CLIENT

def import_ds_module(module_name, compat = True):
    loader = module_name
    use_dask = False

    if compat:
        if module_name == "pandas":
            loader = "dask.dataframe"
            use_dask = True

        if module_name.startswith("sklearn"):
            loader = module_name.replace("sklearn", "dask_ml")
            use_dask = True

    try:
        if use_dask:
            _get_dask_client()

        import_handler(loader)

        logger.success("Successfully imported module '%s'." % loader)
    except (AttributeError, ImportError):
        logger.warn("Unable to import %s, using %s instead", loader, module_name)

    return import_handler(module_name)