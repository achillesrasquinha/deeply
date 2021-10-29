from deeply.model.factory import ModelFactory

class BackBone(ModelFactory):
    def __init__(self):
        self._model = None

    def get(self, name, *args, **kwargs):
        self._model = ModelFactory.get(name, *args, **kwargs)