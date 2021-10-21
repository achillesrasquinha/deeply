import numpy as np
import cv2

from imgaug.augmenters import Augmenter

# https://github.com/aleju/imgaug/issues/128
class UnsharpMask(Augmenter):
    def __init__(self, *args, **kwargs):
        self._super = super(UnsharpMask, self)
        self._super.__init__(*args, **kwargs)
        
    def _augment_batch_(self, batch, *args, **kwargs):
        if batch.images is None:
            return batch

        images = batch.images

        for i, image in enumerate(images):
            pass
        
        return batch

    def get_parameters(self):
        return []