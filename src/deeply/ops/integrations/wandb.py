import os.path as osp

from bpyutils.util.environ import setenv

from deeply.config import PATH
from deeply.ops.integrations.base import BaseService

_PREFIX = "WANDB"

class WeightsAndBiases(BaseService):
    module = "wandb"

    def __init__(self, *args, **kwargs):
        self._super = super(*args, **kwargs)
        self._super.__init__(*args, **kwargs)

        setenv("API_KEY", self.api_key, prefix = _PREFIX)
        setenv("DIR", PATH["CACHE"], prefix = _PREFIX)

    def init(self, name):
        module = self.module

        self._runner = module.init(project = name)

    def upload(self, *files):
        pass

    def watch(self, *models):
        module = self.module
        WandbCallback = module.keras.WandbCallback
        
        for model in models:
            model.add_callback(WandbCallback())