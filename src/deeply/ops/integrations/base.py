import re
from collections import Mapping

from bpyutils.util.imports import import_or_raise
from bpyutils.util.environ import getenv, getenvvar

from deeply.__attr__ import __name__ as NAME

_PREFIX = NAME.upper()

class BaseService:
    def __init__(self, module = None, api_key = None):
        module = getattr(self, "module", module)

        if not module:
            raise ValueError("module not defined.")
        else:
            namespace = module
            registry  = None

            if isinstance(namespace, Mapping):
                namespace = namespace["namespace"]
                registry  = namespace["registry"]
            
            self.module = import_or_raise(namespace, name = registry)

        self.api_key = getenv("API_KEY", prefix = self._get_environ_prefix(), default = api_key)

    def _get_environ_prefix(self):
        klass       = self.__class__
        class_name  = klass.__name__

        suffix = re.sub(r'(?P<name>[A-Z])', '_\g<name>', class_name)
        suffix = suffix.upper()

        return "%s%s" % (_PREFIX, suffix)

    def init(self, name):
        raise NotImplementedError

    def upload(self):
        raise NotImplementedError

    def watch(self):
        raise NotImplementedError