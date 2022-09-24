import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.raw_ops import BatchDataset

from deeply.const import DEFAULT
from deeply.model.layer import ConvBlock
from deeply.model.generate import GenerativeModel
from deeply.model.autoencoder import AutoEncoder
from deeply.callbacks import GANPlotCallback
from deeply.util.model import create_model_fn, update_kwargs

from deeply.datasets.util import length as dataset_length

from bpyutils.util._dict import merge_dict

_binary_cross_entropy_logits = BinaryCrossentropy(from_logits = True)

def generator_loss(fake_output):
    return _binary_cross_entropy_logits(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss  = _binary_cross_entropy_logits(tf.ones_like(real_output),  real_output)
    fake_loss  = _binary_cross_entropy_logits(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss

class GANModel(AutoEncoder):
    @property
    def generator(self):
        return self.decoder

    @property
    def discriminator(self):
        return self.encoder

    def fit(self, *args, **kwargs):
        # kwargs = update_kwargs(kwargs, {
        #     "callbacks": {
        #         "default": [],
        #         "item": GANPlotCallback(self)
        #     }
        # })

        super_ = super(GANModel, self)
        return super_.fit(*args, **kwargs)

    def compute_loss(self, data):
        generator_input_shape = self.generator.input.shape
        _, noise_dim  = generator_input_shape

        batch_size    = tf.shape(data)[0]

        noise = tf.random.normal(shape = (batch_size, noise_dim))

        generated_output = self.generator(noise, training = True)
        
        real_output      = self.discriminator(data, training = True)
        fake_output      = self.discriminator(generated_output, training = True)

        loss_generator     = generator_loss(fake_output)
        loss_discriminator = discriminator_loss(real_output, fake_output)
        
        return loss_generator, loss_discriminator

    def train_step(self, data):
        generator = self.generator
        discriminator = self.discriminator

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            loss_generator, loss_discriminator = self.compute_loss(data)

        generator_gradients = generator_tape.gradient(loss_generator, self.generator.trainable_variables)
        discriminator_gradients = discriminator_tape.gradient(loss_discriminator, self.discriminator.trainable_variables)
        
        self.optimizers["decoder"].apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.optimizers["encoder"].apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        return merge_dict(self.compute_metrics(), {
            "generator-loss": loss_generator,
            "discriminator-loss": loss_discriminator,
            "loss": loss_generator + loss_discriminator
        })

    def compute_metrics(self):
        generator = self.generator
        discriminator = self.discriminator

        # TODO: update metrics
        return merge_dict(
            { "%s-%s" % (generator.name, m.name): m.result() for m in generator.metrics },
            { "%s-%s" % (discriminator.name, m.name): m.result() for m in discriminator.metrics }
        )

GAN = create_model_fn(
    func = GenerativeModel,
    doc  = \
    """

    """,
    args = {
        "name": "gan",
        "model_type": GANModel,
        "encoder_name": "discriminator",
        "decoder_name": "generator"
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