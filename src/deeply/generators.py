import os, os.path as osp

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

from deeply.const import DEFAULT
from deeply.util.system import get_basename

IMAGE_MASK_GENERATOR_SIZE = (256, 256)

class BaseDataGenerator(Sequence):
    def __init__(self,
        X = None,
        batch_size = DEFAULT["batch_size"],
        shuffle = False
    ):
        self.batch_size  = batch_size
        self._n_samples  = len(X or [])
        self._shuffle    = shuffle

    @property
    def n_samples(self):
        return getattr(self, "_n_samples", 0)

    def __len__(self):
        return int(np.floor(self.n_samples) / self.batch_size)

def _get_file_generator(dir_path):
    dir_path = osp.abspath(dir_path)
    files    = os.listdir(dir_path)
    for file in files:
        yield osp.join(dir_path, file)

def _load_img_arr(*args, **kwargs):
    return img_to_array(load_img(*args, **kwargs))

class ImageMaskGenerator(BaseDataGenerator):
    """
    Image Mask Data Generator
    """
    def __init__(self,
        dir_images,
        dir_masks,
        color_mode = "rgb",
        image_size = IMAGE_MASK_GENERATOR_SIZE,
        mask_size  = IMAGE_MASK_GENERATOR_SIZE,
        *args, **kwargs
    ):
        self.super = super(ImageMaskGenerator, self)
        self.super.__init__(*args, **kwargs)

        self.dir_images = osp.abspath(dir_images)
        self.dir_masks  = osp.abspath(dir_masks)

        self.image_size    = image_size
        self.mask_size     = mask_size

        self.color_mode    = color_mode

        self.on_epoch_end()

    @property
    def image_files(self):
        return _get_file_generator(self.dir_images)

    @property
    def mask_files(self):
        return _get_file_generator(self.dir_masks)

    @property
    def n_samples(self):
        return len(list(self.image_files))

    def __getitem__(self, index):
        batch_size = self.batch_size
        color_mode = self.color_mode

        from_, to  = index * batch_size, (index + 1) * batch_size
        
        filenames  = self.indexes[from_:to]

        n_channels = 1 if color_mode == "grayscale" else 3

        images = np.empty((batch_size, *self.image_size, n_channels))
        masks  = np.empty((batch_size, *self.mask_size,  n_channels))

        for i, filename in enumerate(filenames):
            image_path = osp.join(self.dir_images, filename)
            mask_path  = osp.join(self.dir_masks,  filename)

            image = _load_img_arr(path = image_path, color_mode = self.color_mode,
                target_size = self.image_size)
            mask  = _load_img_arr(path = mask_path,  color_mode = self.color_mode,
                target_size = self.mask_size)
            
            image = image / 255
            mask  = mask  / 255

            mask[mask >  0.5] = 1
            mask[mask <= 0.5] = 0

            images[i] = image
            masks[i]  = mask

        return images, masks
    
    def on_epoch_end(self):
        self.indexes = np.asarray([get_basename(o) for o in self.image_files])
        if self._shuffle:
            np.random.shuffle(self.indexes)