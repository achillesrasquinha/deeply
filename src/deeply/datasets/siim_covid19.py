import os.path as osp
from glob import glob
import csv

import tensorflow as tf
from tensorflow.io.gfile import (
    GFile
)
import tensorflow_datasets as tfds
from tensorflow_datasets.core import (
    Version,
    GeneratorBasedBuilder,
    DatasetInfo,
    SplitGenerator
)
from tensorflow_datasets.core.features import (
    FeaturesDict,
    Image as ImageF,
    Text,
    Tensor,
    ClassLabel,
)
import numpy as np
from PIL import Image
import imageio

from deeply.util.string import strip, safe_decode
from deeply.util.system import makedirs
from deeply._compat import iterkeys, iteritems

_DATASET_KAGGLE      = "siim-covid19-detection"
_DATASET_HOMEPAGE    = ""
_DATASET_DESCRIPTION = """
"""
_DATASET_CITATION    = """\
"""

_SANITIZE_LABELS = {
    "sex": {
        "male": ["M"],
        "female": ["F"],
        "other": ["O"]
    }
}

def _sanitize_label(type_, label):
    type_ = _SANITIZE_LABELS[type_]
    
    for key, value in iteritems(type_):
        if label in value:
            return key

    return label

def sanitize_lines(lines):
    return list(filter(bool, [strip(line) for line in lines]))

def _str_to_int(o):
    stripped = "".join((s for s in o if s.isdigit()))
    stripped = stripped.lstrip("0")
    
    return int(stripped)

def merge_images(*args, **kwargs):
    assert len(args) >= 2

    arr = imageio.imread(args[0])

    for path in args:
        a   = imageio.imread(path)
        arr = np.maximum(arr, a)

    output = kwargs.get("output")

    if output:
        img = Image.fromarray(arr)
        img.save(output)

    return arr

class SIIMCOVID19(GeneratorBasedBuilder):
    """
    SIIM COVID-19 Chest X-ray Dataset.
    """

    VERSION = Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial Release."
    }

    def _info(self):
        return DatasetInfo(
            builder     = self,
            description = _DATASET_DESCRIPTION,
            features    = FeaturesDict({
                "image": ImageF(encoding_format = "png"),
                 "mask": ImageF(encoding_format = "png"),
                  "sex": ClassLabel(names = iterkeys(_SANITIZE_LABELS["sex"])),
                  "age": Tensor(shape = (), dtype = tf.uint8),
                "label": Text(),
            }),
            supervised_keys = ("image", "mask"),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_extracted = dl_manager.download_kaggle_data(_DATASET_KAGGLE)

        return [
            SplitGenerator(
                name = tfds.Split.TRAIN,
                gen_kwargs = {
                    "csv_path": osp.join(path_extracted, "train_image_level.csv")
                }
            ),
            SplitGenerator(
                name = tfds.Split.TEST,
                gen_kwargs = {
                    "csv_path": osp.join(path_extracted, "test_image_level.csv")
                }
            )
        ]
        
    def _generate_examples(self, csv_path):
        with GFile(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                

        # path_images = osp.join(path, "CXR_png")
        # path_data   = osp.join(path, "ClinicalReadings")
        # path_masks  = osp.join(path, "ManualMask")
        # path_masks_merged = osp.join(path_masks, "merged")

        # makedirs(path_masks_merged, exist_ok = True)

        # for path_img in glob(osp.join(path_images, "*.png")):
        #     fname  = osp.basename(osp.normpath(path_img))
        #     prefix = str(fname).split(".png")[0]

        #     path_txt  = osp.join(path_data, "%s.txt" % prefix)

        #     path_mask = osp.join(path_masks_merged, "%s.png" % prefix)

        #     if not osp.exists(path_mask):
        #         path_mask_left  = osp.join(path_masks, "leftMask",  "%s.png" % prefix)
        #         path_mask_right = osp.join(path_masks, "rightMask", "%s.png" % prefix)

        #         merge_images(path_mask_left, path_mask_right, output = path_mask)
                
        #     with open(path_txt) as f:
        #         content = f.readlines()
        #         lines   = sanitize_lines(content)

        #         sex     = _sanitize_label("sex", safe_decode(strip(lines[0].split(": ")[1])))
        #         age     = _str_to_int(safe_decode(strip(lines[1].split(": ")[1])))
        #         label   = safe_decode(strip(lines[2]))

        #         yield prefix, {
        #             "image": path_img,
        #              "mask": path_mask,
        #               "sex": sex,
        #               "age": age,
        #             "label": label
        #         }