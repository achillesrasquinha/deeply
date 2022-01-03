from bpyutils.util.environ import setenv

from deeply.config import PATH
from deeply.ops.integrations.base import BaseService
from deeply.exception import OpsError

_PREFIX = "KAGGLE"

class Kaggle(BaseService):
    module = "kaggle"

    def __init__(self, *args, **kwargs):
        self._super = super(Kaggle, self)
        self._super.__init__(*args, **kwargs)

        setenv("API_KEY", self.api_key, prefix = _PREFIX)
        setenv("DIR", PATH["CACHE"], prefix = _PREFIX)

    def upload(self, *files, name = None, dest = None):
        pass

    def download(self, name, target_dir = None):
        pass