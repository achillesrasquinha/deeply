from bpyutils.util.environ import setenv

from deeply.ops.integrations.base import BaseService

class WeightsAndBiases(BaseService):
    module = "wandb"

    def init(self, name):
        module = self.module
        setenv("API_KEY", self.api_key, prefix = "WANDB")

        self._runner = module.init(project = name)

    def upload(self, *files):
        pass

    def watch(self, models):
        module = self.module
        WandbCallback = module.keras.WandbCallback

        