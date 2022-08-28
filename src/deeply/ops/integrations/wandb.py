import os.path as osp

from bpyutils.util.environ import setenv
from bpyutils.util.string  import get_random_str

from deeply.config import PATH
from deeply.ops.integrations.base import BaseService
from deeply.exception import OpsError

from bpyutils._compat import Mapping

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
        runner   = self._runner

        name     = name or get_random_str()
        aliases  = "latest"

        if ":" in name:
            name, aliases = name.split(":")

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

        runner.log_artifact(artifact, aliases = aliases)

    def download(self, name, target_dir = None):
        module      = self.module
        runner      = self._runner

        target_dir  = target_dir or PATH["CACHE"]

        try:
            artifact  = runner.use_artifact(name)
            directory = artifact.download(root = target_dir)
        except (module.errors.CommError, ValueError):
            raise OpsError("No data object %s found." % name)

        return directory

    def watch(self, *models):
        module = self.module
        WandbCallback = module.keras.WandbCallback
        
        for model in models:
            model.add_callback(WandbCallback())