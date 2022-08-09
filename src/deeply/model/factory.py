from bpyutils.util.types   import lmap
from bpyutils.util.imports import import_handler

class ModelFactory:
    MODELS = {
        **{
            key: value for key, value in lmap(lambda x: ("efficient-net-b%s" % x, {
                "model_class": import_handler("tensorflow.keras.applications.EfficientNetB%s" % x)
            }), range(8))

        },
        "variational-autoencoder": {
            "model_class": import_handler("deeply.model.vae.VAE")
        },
        "convolutional-variational-autoencoder": {
            "model_class": import_handler("deeply.model.vae.ConvolutionalVAE")
        },
        "generative-adversarial-network": {
            "model_class": import_handler("deeply.model.gan.GAN")
        },
        "convolutional-generative-adversarial-network": {
            "model_class": import_handler("deeply.model.gan.DCGAN")
        }
    }

    def get(name, *args, **kwargs):
        if name not in ModelFactory.MODELS:
            raise ValueError("No model %s found." % name)

        model = ModelFactory.MODELS[name]
        model_class = model["model_class"]

        return model_class(*args, **kwargs)