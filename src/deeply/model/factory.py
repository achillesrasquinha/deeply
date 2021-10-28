from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7
)

class ModelFactory:
    MODELS = {
        "efficient-net-b0": {
            "model_class": EfficientNetB0
        },
        "efficient-net-b1": {
            "model_class": EfficientNetB1
        },
        "efficient-net-b2": {
            "model_class": EfficientNetB2
        },
        "efficient-net-b3": {
            "model_class": EfficientNetB3
        },
        "efficient-net-b4": {
            "model_class": EfficientNetB4
        },
        "efficient-net-b5": {
            "model_class": EfficientNetB5
        },
        "efficient-net-b6": {
            "model_class": EfficientNetB6
        },
        "efficient-net-b7": {
            "model_class": EfficientNetB7
        }
    }

    def get(self, name, *args, **kwargs):
        if name not in ModelFactory.MODELS:
            raise ValueError("No model %s found." % name)

        model = ModelFactory.MODELS[name]
        model_class = model["model_class"]

        return model_class(*args, **kwargs)