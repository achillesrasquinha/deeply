from deeply.ops.integrations.base import BaseService

class AzureML(BaseService):
    module = "azureml"

    def init(self, name):
        module      = self.module
        Workspace   = module.core.workspace.Workspace

        self._ws    = Workspace(
            name = name
        )

    def upload(self):
        pass