import os.path as osp

import numpy as np
import imageio
from PIL import Image
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from bpyutils.util.system import makedirs, get_basename
from bpyutils.util.string import get_random_str

def to_binary(arr, threshold = None, min_ = 0, max_ = 1):
    arr = np.asarray(arr)

    if threshold is None:
        threshold = np.amax(arr) / 2

    arr[arr <  threshold] = min_
    arr[arr >= threshold] = max_

    return arr

def augment(augmentor, images, masks = None, dir_images = None, dir_masks = None, prefix = None, filename = None, format_ = "jpg"):
    segmaps = None

    if masks is not None:
        segmaps = [SegmentationMapsOnImage(to_binary(mask), shape = mask.shape) for mask in masks]
    
    images, segmaps = augmentor(images = images, segmentation_maps = segmaps)

    if filename:
        dir_images = osp.dirname(filename)
        filename   = get_basename(filename)

    makedirs(dir_images, exist_ok = True)

    if segmaps is not None:
        makedirs(dir_masks, exist_ok = True)
    
    for i, (image, segmap) in enumerate(zip(images, segmaps)):

        suffix = i if prefix else ""

        fname  = filename or "%s%s.%s" % (prefix or get_random_str(), suffix, format_)
        path   = osp.join(dir_images, fname)
        
        imageio.imwrite( path, image )

        if segmap is not None:
            canvas  = np.zeros(image.shape, dtype = np.uint8)
            mask    = segmap.draw_on_image(canvas)[0]

            path    = osp.join(dir_masks, fname)

            image   = Image.fromarray(mask)
            image   = image.convert("L")

            array   = np.asarray(image)
            array   = to_binary(array, max_ = 255)

            imageio.imwrite( path, array )