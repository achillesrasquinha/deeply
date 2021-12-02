from deeply.model.factory import ModelFactory

from bpyutils.util.types import lmap

_EFFICIENT_NET_FEATURE_LAYERS = [
    "block2a_expand_activation",
    "block3a_expand_activation",
    "block4a_expand_activation",
    "block6a_expand_activation"
]

MODELS = {
    key: value for key, value in lmap(lambda x: ("efficient-net-b%s" % x, {
            "feature_layers": _EFFICIENT_NET_FEATURE_LAYERS
        }
    ), range(8))
}

class BackBone(ModelFactory):
    def __init__(self, name, *args, **kwargs):
        if name not in MODELS:
            raise ValueError("No backbone %s found." % name)

        self._name = name
        self._build(*args, **kwargs)

    def _build(self, *args, **kwargs):
        self._model = ModelFactory.get(self._name, include_top = False, *args, **kwargs)

    def get_feature_layers(self):
        model_meta = MODELS[self._name]
        return [self._model.get_layer(feature_layer) for feature_layer in model_meta["feature_layers"]]