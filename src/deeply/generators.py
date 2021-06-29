import os, os.path as osp
import itertools

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from deeply.const import DEFAULT
from deeply.util._dict  import merge_dict
from deeply.util.system import pardir

IMAGE_MASK_GENERATOR_SIZE = (256, 256)

class BaseDataGenerator(Sequence):
    def __init__(self,
        X = None,
        batch_size = DEFAULT["batch_size"]
    ):
        self.batch_size = batch_size
        self.n_samples  = len(X or [])

    def __len__(self):
        return int(np.floor(self.n_samples) / self.batch_size)

# class ImageMaskGenerator(BaseDataGenerator):
#     def __init__(self,
#         dir_images,
#         dir_masks,
#         *args, **kwargs
#     ):
#         self.super = super(ImageMaskGenerator, self)

#         self.dir_images = dir_images
#         self.dir_masks  = dir_masks

#         self.super.__init__(*args, **kwargs)

#     def __len__(self):
#         self.n_samples = len(os.listdir(self.dir_images))
#         return self.super.__len__()

#     def __getitem__(self, key):
#         datagen_o_image = ImageDataGenerator()
#         datagen_o_mask  = ImageDataGenerator()

#         datagen_image   = datagen_o_image.flow_from_directory(self.dir_images)
#         datagen_mask    = datagen_o_mask.flow_from_directory(self.dir_masks)

#         for (image, mask) in zip(datagen_image, datagen_mask):
#             image = image / 255
#             mask  = mask  / 255

#             mask[mask >  0.5] = 1
#             mask[mask <= 0.5] = 0

#             yield (image, mask)

def get_basename(path):
    return osp.basename(osp.normpath(path))

def ImageMaskGenerator(dir_images, dir_masks, **kwargs):
    datagen_o_image = ImageDataGenerator()
    datagen_o_mask  = ImageDataGenerator()

    seed = kwargs.get("seed", 1)
    mode = kwargs.get("mode", "rgb")
    batch_size = kwargs.get("batch_size", 32)
    size_image = kwargs.get("image_size", IMAGE_MASK_GENERATOR_SIZE)
    size_mask  = kwargs.get("mask_size",  IMAGE_MASK_GENERATOR_SIZE)

    fn_args = dict(class_mode = None, color_mode = mode,
        batch_size = batch_size, seed = seed)

    dir_par     = osp.abspath(pardir(dir_images))

    class_images = get_basename(dir_images)
    class_masks  = get_basename(dir_masks)
    
    datagen_image  = datagen_o_image.flow_from_directory(dir_par, **merge_dict(fn_args,
        dict(target_size = size_image, classes = [class_images])))
    datagen_mask    = datagen_o_mask.flow_from_directory(dir_par, **merge_dict(fn_args,
        dict(target_size = size_mask,  classes = [class_masks])))

    data_generator = zip(datagen_image, datagen_mask)

    for (image, mask) in data_generator:
        image = image / 255
        mask  = mask  / 255

        mask[mask >  0.5] = 1
        mask[mask <= 0.5] = 0

        yield (image, mask)