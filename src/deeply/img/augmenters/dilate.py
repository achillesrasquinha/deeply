import numpy as np
import cv2

from imgaug.augmenters import Augmenter

# https://github.com/aleju/imgaug/issues/128
class Dilate(Augmenter):
    def __init__(self, kernel = np.ones((1, 1)), iterations = 1, *args, **kwargs):
        self._super = super(Dilate, self)
        self._super.__init__(*args, **kwargs)
        
        self.kernel     = kernel
        self.iterations = iterations
        
    def _augment_batch_(self, batch, *args, **kwargs):
        if batch.images is None:
            return batch

        images = batch.images

        for i, image in enumerate(images):
            batch.images[i] = cv2.dilate(image, self.kernel, iterations = self.iterations)

        return batch

    def get_parameters(self):
        return [self.kernel, self.iterations]