from bpyutils.util.imports import import_handler
from bpyutils.log import get_logger

from deeply.__attr__ import __name__ as NAME

logger = get_logger(NAME)

def import_ds_module(module_name, compat = True):
    loader = module_name

    if compat:
        if module_name == "pandas":
            loader = "dask.dataframe"

        if module_name.startswith("sklearn"):
            loader = module_name.replace("sklearn", "dask_ml")

    try:
        import_handler(loader)
    except ImportError:
        logger.warn("Unable to import %s, using %s instead", loader, module_name)

    return import_handler(module_name)