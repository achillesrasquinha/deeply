import os.path as osp
from collections import Mapping

from bpyutils.util.environ import setenv
from bpyutils.util.string  import get_random_str

from deeply.config import PATH
from deeply.ops.integrations.base import BaseService

_PREFIX = "WANDB"

_ADD_FILE = "add_file"
_ADD_DIR  = "add_dir"

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

    def upload(self, *files, name = None, dest = None):
        module   = self.module
        name     = name or get_random_str()

        artifact = module.Artifact(name, type = 'dataset')

        for f in files:
            source      = f
            destination = dest

            if isinstance(f, Mapping):
                source      = f["source"]
                destination = f["destination"]

                module_attr = _ADD_FILE if osp.isfile(source) else _ADD_DIR
            elif osp.isfile(f):
                module_attr = _ADD_FILE
            elif osp.isdir(f):
                module_attr = _ADD_DIR
                
            fn = getattr(artifact, module_attr)
            fn(source, destination)

        self._runner.log_artifact(artifact)

    def watch(self, *models):
        module = self.module
        WandbCallback = module.keras.WandbCallback
        
        for model in models:
            model.add_callback(WandbCallback())