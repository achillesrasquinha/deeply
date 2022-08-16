import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam


from tensorflow.python.data.ops.dataset_ops import BatchDataset

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

    def compile(self, *args, **kwargs):
        optimizer     = kwargs.pop("optimizer", Adam)
        learning_rate = kwargs.pop("learning_rate", None)

        generator_learning_rate     = learning_rate or kwargs.get("generator_learning_rate",     DEFAULT["generative_model_encoder_learning_rate"])
        discriminator_learning_rate = learning_rate or kwargs.get("discriminator_learning_rate", DEFAULT["generative_model_decoder_learning_rate"])

        self._optimizer = {}
        self._optimizer["generator"]     = optimizer(learning_rate = generator_learning_rate)
        self._optimizer["discriminator"] = optimizer(learning_rate = discriminator_learning_rate)

        self._super.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        kwargs = update_kwargs(kwargs, {
            "callbacks": {
                "default": [],
                "item": GANPlotCallback(self)
            }
        })

        args = list(args)
        
        X = args[0]

        n_samples = dataset_length(X)
        self._batch_size = kwargs.get("batch_size", None)

        if self._batch_size:
            args[0] = X.batch(self._batch_size)
            kwargs["steps_per_epoch"] = n_samples / self._batch_size

        return self._super.fit(*args, **kwargs)

    @tf.function
    def train_step(self, data):
        generator_input_shape = self.generator.input.shape
        _, noise_dim  = generator_input_shape

        generator     = self.generator
        discriminator = self.discriminator

        noise = tf.random.normal([self._batch_size or 1, noise_dim]) # TODO: get batch size from data

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated_output = generator(noise, training = True)

            real_output = discriminator(data, training = True)
            fake_output = discriminator(generated_output, training = True)

            loss_generator      = generator_loss(fake_output)
            loss_discriminator  = discriminator_loss(real_output, fake_output)

        generator_train_vars     = generator.trainable_variables
        discriminator_train_vars = discriminator.trainable_variables

        generator_gradients      = generator_tape.gradient(loss_generator, generator_train_vars)
        discriminator_gradients  = discriminator_tape.gradient(loss_discriminator, discriminator_train_vars)
        
        self._optimizer["generator"].apply_gradients(zip(generator_gradients, generator_train_vars))
        self._optimizer["discriminator"].apply_gradients(zip(discriminator_gradients, discriminator_train_vars))

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