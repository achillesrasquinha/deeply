from bpyutils.util.types   import lmap
from bpyutils.util.imports import import_handler

class ModelFactory:
    MODELS = {
        **{
            key: value for key, value in lmap(lambda x: ("efficient-net-b%s" % x, {
                "model_class": import_handler("tensorflow.keras.applications.EfficientNetB%s" % x)
            }), range(8))

        },
        "vae": {
            "model_class": import_handler("deeply.model.vae.VAE")
        },
        "cvae": {
            "model_class": import_handler("deeply.model.vae.ConvolutionalVAE")
        },
        "gan": {
            "model_class": import_handler("deeply.model.gan.GAN")
        },
        "dcgan": {
            "model_class": import_handler("deeply.model.gan.DCGAN")
        },
        "mlp": {
             "model_class": import_handler("deeply.model.mlp.MLP")
        }
    }

    def get(name, *args, **kwargs):
        if name not in ModelFactory.MODELS:
            raise ValueError("No model %s found." % name)

        model = ModelFactory.MODELS[name]
        model_class = model["model_class"]

        return model_class(*args, **kwargs)