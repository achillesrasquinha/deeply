import tensorflow as tf
from tensorflow.keras.callbacks import Callback

import matplotlib.pyplot as pplt

from bpyutils.log import get_logger

from deeply.__attr__ import __name__ as NAME
from deeply.const import DEFAULT
from deeply.plots import imgplot

logger = get_logger(NAME)

class GANPlotCallback(Callback):
    def __init__(self, model, n_samples = DEFAULT["gan_plot_callback_samples"], *args, **kwargs):
        self._super = super(GANPlotCallback, self)
        self._super.__init__(*args, **kwargs)

        self.n_samples   = n_samples

        generator        = model.generator
        generator_input_shape = generator.input_shape
        _, noise_dim = generator_input_shape

        self.sample_data = tf.random.normal([self.n_samples, noise_dim])

    def on_train_batch_end(self, batch, logs = None):
        try:
            # from IPython import display

            # display.clear_output(wait = True)

            generated_images = self.model.generator(self.sample_data, training = False)

            if self.model.scaler:
                generated_images = self.model.scaler.inverse_transform(generated_images)
                
            generated_images = tf.cast(generated_images, tf.uint8)

            imgplot(generated_images, to_file = "gan.png", cmap = "gray")
        except ImportError:
            logger.warn("IPython is not installed. Skipping...")