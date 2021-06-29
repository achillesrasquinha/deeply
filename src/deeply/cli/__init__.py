# imports - module imports
from deeply.cli.util   import *
from deeply.cli.parser import get_args
from deeply.util._dict import merge_dict
from deeply.util.types import get_function_arguments

def command(fn):
    args    = get_args()
    
    params  = get_function_arguments(fn)

    params  = merge_dict(params, args)
    
    def wrapper(*args, **kwargs):
        return fn(**params)

    return wrapper