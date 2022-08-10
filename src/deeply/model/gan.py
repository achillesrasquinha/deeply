import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Layer,
    Activation,
    Conv2DTranspose,
    Flatten,
    Reshape,
    Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from deeply.util.model      import get_input
from deeply.model.base      import BaseModel
from deeply.model.layer     import ConvBlock, DenseBlock, Conv2DTransposeBlock
from deeply.model.types     import is_layer_type
from deeply.model.autoencoder import AutoEncoder
from deeply.callbacks import GANPlotCallback
from deeply.const import DEFAULT
from deeply.util.model import create_model_fn
from deeply.model.generate import GenerativeModel

from bpyutils.util._dict    import merge_dict
from bpyutils.util.imports  import import_handler
from bpyutils.util.array    import sequencify

_binary_cross_entropy_logits = BinaryCrossentropy(from_logits = True)

def generator_loss(fake_output):
    return _binary_cross_entropy_logits(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = _binary_cross_entropy_logits(tf.ones_like(real_output),  real_output)
    fake_loss = _binary_cross_entropy_logits(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss

class GANModel(AutoEncoder):
    def __init__(self, encoder, decoder, *args, **kwargs):
        self._super = super(GANModel, self)
        self._super.__init__(encoder, decoder, *args, **kwargs)

    @property
    def generator(self):
        return self.decoder

    @property
    def discriminator(self):
        return self.encoder

    def compile(self, *args, **kwargs):
        kwargs["loss"] = kwargs.get("loss", _binary_cross_entropy_logits)

        metrics = sequencify(kwargs.get("metrics", []))
        metrics.append(_binary_cross_entropy_logits)

        kwargs["metrics"] = metrics

        learning_rate     = kwargs.pop("learning_rate", DEFAULT["gan_learning_rate"])

        self.generator_optimizer     = Adam(learning_rate)
        self.discriminator_optimizer = Adam(learning_rate)

        self._super.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        callbacks = kwargs.pop("callbacks", [])
        callbacks.append(GANPlotCallback(self))

        kwargs["callbacks"] = callbacks

        return self._super.fit(*args, **kwargs) 

    def train_step(self, data):
        generator_input_shape = self.generator.input.shape
        _, noise_dim = generator_input_shape

        noise = tf.random.normal([1, noise_dim])

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated_output = self.generator(noise, training = True)

            real_output = self.discriminator(data, training = True)
            fake_output = self.discriminator(generated_output, training = True)

            loss_generator      = generator_loss(fake_output)
            loss_discriminator  = discriminator_loss(real_output, fake_output)

        generator_train_vars     = self.generator.trainable_variables
        discriminator_train_vars = self.discriminator.trainable_variables

        generator_gradients     = generator_tape.gradient(loss_generator, generator_train_vars)
        discriminator_gradients = discriminator_tape.gradient(loss_discriminator, discriminator_train_vars)
        
        self.generator_optimizer.apply_gradients(zip(generator_gradients, generator_train_vars))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_train_vars))

        # self.compiled_metrics.update_state(data, y_pred)

        return { m.name: m.result() for m in self.metrics }

GAN   = create_model_fn(
    func = GenerativeModel,
    doc  = \
    """

    """,
    args = {
        "name": "gan",
        "model_type": GANModel
    }
)


DCGAN = create_model_fn(
    func = GAN,
    doc  = \
    """
    Constructs a Deep Convolutional Generative Adversarial Network.

    References
        [1]. https://arxiv.org/abs/1511.06434 
        
    >>> import deeply
    >>> model = deeply.hub("dcgan")
    """,
    args = {
        "name": "dcgan",
        "layer_block": ConvBlock
    }
)