<<<<<<< HEAD
# imports - standard imports
import subprocess as sp

class DeeplyError(Exception):
    pass

class PopenError(DeeplyError, sp.CalledProcessError):
    pass

class DependencyNotFoundError(ImportError):
    pass

class OpsError(DeeplyError):
=======
class DeeplyError(Exception):
    pass

class DependencyNotFoundError(ImportError):
>>>>>>> template/master
    pass