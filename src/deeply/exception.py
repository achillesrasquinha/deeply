# imports - standard imports
import subprocess as sp

class DeeplyError(Exception):
    pass

class PopenError(DeeplyError, sp.CalledProcessError):
    pass

class DependencyNotFoundError(ImportError):
    pass

class OpsError(DeeplyError):
    pass