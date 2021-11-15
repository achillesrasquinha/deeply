from deeply.model.factory import ModelFactory

_EFFICIENT_NET_FREEZE_LAYERS = [
    "block6a_expand_activation",
    "block4a_expand_activation",
    "block3a_expand_activation",
    "block2a_expand_activation"
]

class BackBone(ModelFactory):
    MODELS = {
        "efficient-net-b0": {
            "freeze_layers": _EFFICIENT_NET_FREEZE_LAYERS
        },
        "efficient-net-b1": {
            "freeze_layers": _EFFICIENT_NET_FREEZE_LAYERS
        },
        "efficient-net-b2": {
            "freeze_layers": _EFFICIENT_NET_FREEZE_LAYERS
        },
        "efficient-net-b3": {
            "freeze_layers": _EFFICIENT_NET_FREEZE_LAYERS
        },
        "efficient-net-b4": {
            "freeze_layers": _EFFICIENT_NET_FREEZE_LAYERS
        },
        "efficient-net-b5": {
            "freeze_layers": _EFFICIENT_NET_FREEZE_LAYERS
        },
        "efficient-net-b6": {
            "freeze_layers": _EFFICIENT_NET_FREEZE_LAYERS
        },
        "efficient-net-b7": {
            "freeze_layers": _EFFICIENT_NET_FREEZE_LAYERS
        }
    }

    def __init__(self):
        self._model = None

    def get(self, name, *args, **kwargs):
        self._model = ModelFactory.get(name, *args, **kwargs)