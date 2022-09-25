import os.path as osp

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.raw_ops import BatchDataset
from tensorflow.data import Dataset

from deeply.const import DEFAULT
from deeply.model.layer import ConvBlock
from deeply.model.generate import GenerativeModel
from deeply.model.autoencoder import AutoEncoder
from deeply.callbacks import GANPlotCallback
from deeply.util.model import create_model_fn, update_kwargs
from deeply.datasets.util import length as dataset_length
from deeply.plots import imgplot as imgplt, mplt

from bpyutils.util.system import make_temp_file
from bpyutils.util._dict import merge_dict
from bpyutils.util.array import squash

_binary_cross_entropy_logits = BinaryCrossentropy(from_logits = True)

def generator_loss(fake_output):
    return _binary_cross_entropy_logits(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss  = _binary_cross_entropy_logits(tf.ones_like(real_output),  real_output)
    fake_loss  = _binary_cross_entropy_logits(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss

class GANModel(AutoEncoder):
    def __init__(self, *args, **kwargs):
        disc_steps  = kwargs.pop("disc_steps",  DEFAULT["gan_discriminator_train_steps_offset"])
        grad_weight = kwargs.pop("grad_weight", 0)
        
        super_ = super(GANModel, self)
        super_.__init__(*args, **kwargs)

        self.disc_steps  = disc_steps
        self.grad_weight = grad_weight
        
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

    def get_random_latent_vector(self, n_samples = 1):
        generator_input_shape = self.generator.input.shape
        _, noise_dim = generator_input_shape

        # if not hasattr(data, "shape"):
        #     data = squash(data)
            
        #     if len(data) == 2:
        #         data, y = data
        #     else:
        #         data = data[0]

        # data_shape = data.shape

        # if len(data_shape) == 4:
        #     batch_size = data_shape[0]
        # else:
        #     batch_size = 1

        return tf.random.normal(shape = (n_samples, noise_dim))

    def _compute_gradient_penalty(self, real_output, fake_output):
        """
            Compute the gradient penalty.
        """
        batch_size = tf.shape(real_output)[0]

        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff  = fake_output - real_output
        interpolated = real_output + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)

            pred = self.discriminator(interpolated, training = True)

        grads = gp_tape.gradient(pred, [interpolated])[0]

        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis = [1, 2, 3]))

        gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)

        return gradient_penalty

    def train_step(self, data):
        generator = self.generator
        discriminator = self.discriminator

        disc_loss = self.loss_fn["encoder"]
        gen_loss  = self.loss_fn["decoder"]

        batch_size = tf.shape(data)[0]

        # for _ in range(self.disc_steps):
        noise = self.get_random_latent_vector(n_samples = batch_size)

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated_output = generator(noise, training = True)
        
            real_output = discriminator(data, training = True)
            fake_output = discriminator(generated_output, training = True)

            loss_generator = gen_loss(fake_output)
            loss_discriminator = disc_loss(real_output, fake_output)

            if self.grad_weight:
                loss_discriminator = loss_discriminator + self._compute_gradient_penalty(data, generated_output) * self.grad_weight

        discriminator_gradients = discriminator_tape.gradient(loss_discriminator, discriminator.trainable_variables)
        self.optimizers["encoder"].apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        # noise = self.get_random_latent_vector(n_samples = batch_size)
        # with tf.GradientTape() as generator_tape:
        #     generated_output = self.generator(noise, training = True)

        #     fake_output = self.discriminator(generated_output, training = True)

        #     loss_generator = gen_loss(fake_output)

        generator_gradients = generator_tape.gradient(loss_generator, generator.trainable_variables)
        self.optimizers["decoder"].apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        
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
    Constructs a GAN model.

    References
    ----------
        - https://arxiv.org/abs/1406.2661

    >>> import deeply
    >>> model = deeply.hub("gan")
    """,
    args = {
        "name": "gan",
        "model_type": GANModel,
        "encoder_name": "discriminator",
        "decoder_name": "generator",
        "encoder_loss": discriminator_loss,
        "decoder_loss": generator_loss,
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

WGANGP = create_model_fn(
    func = GAN,
    doc  = \
    """
    Constructs a Wasserstein Generative Adversarial Network (with Gradient Penalty).

    References
        [1]. https://arxiv.org/abs/1701.07875 
        
    >>> import deeply
    >>> model = deeply.hub("wcgan-gp")
    """,
    args = {
        "name": "wcgan-gp",
        "grad_weight": DEFAULT["gan_gradient_penalty_weight"]
    }
)

WDCGANGP = create_model_fn(
    func = WGANGP,
    doc  = \
    """
    Constructs a Wasserstein Deep Convolutional Generative Adversarial Network (with Gradient Penalty).

    References
        [1]. https://arxiv.org/abs/1701.07875 
        
    >>> import deeply
    >>> model = deeply.hub("wdcgan-gp")
    """,
    args = {
        "name": "wdcgan-gp",
        "layer_block": ConvBlock
    }
)

def save_img_samples(model, *args, **kwargs):
    samples = kwargs.pop("samples", None)
    to_file = kwargs.pop("to_file", None)
    show_plot = kwargs.pop("show_plot", False)

    if samples != None:
        n_samples = kwargs.pop("n_samples", 16)
        samples = model.get_random_latent_vector(n_samples = n_samples)

    imgs = model.generator(samples, training = False)

    scaler = kwargs.pop("scaler", model.scaler or None)

    if scaler:
        imgs = scaler.inverse_transform(imgs)

    imgs = tf.cast(imgs, tf.uint8)

    if to_file:
        with make_temp_file(fname = "model.png") as tmp_file:
            imgplt(imgs, to_file = tmp_file, *args, **kwargs)

            frames = []

            if osp.exists(to_file):
                with Image.open(to_file) as gif_img:
                    n_frames = gif_img.n_frames
                    for i in range(n_frames - 1, -1, -1):
                        gif_img.seek(i)
                        frames.append(gif_img.copy())

            image = Image.open(tmp_file)
            image.save(fp = to_file, format = "GIF", append_images = frames, 
                save_all = True, duration = 100, loop = 0)

    if show_plot:
        mplt.show()