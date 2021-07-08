import os.path as osp

import imageio

from deeply.util.system import makedirs, get_basename
from deeply.util.string import get_random_str

def augment(augmentor, images, dir_path = None, filename = None, format_ = "jpg"):
    images = augmentor(images = images)

    if filename:
        dir_path = osp.dirname(filename)
        filename = get_basename(filename)

    makedirs(dir_path, exist_ok = True)
    
    for image in images:
        fname = filename or "%s.%s" % (get_random_str(), format_)
        path  = osp.join(dir_path, fname)
        imageio.imwrite( path, image )