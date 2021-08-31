import os, os.path as osp
from glob import glob
import csv

import numpy as np

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
import pydicom

from deeply.util.string import strip, safe_decode
from deeply.util.system import makedirs
from deeply._compat import iterkeys, iteritems

_DATASET_KAGGLE      = "rsna-miccai-brain-tumor-radiogenomic-classification"
_DATASET_HOMEPAGE    = ""
_DATASET_DESCRIPTION = """
"""
_DATASET_CITATION    = """\
"""

def dicom_to_image_file(dicom, path_file):
    # preprocess
    array = dicom.pixel_array
    array = array.astype(np.float64)
    array = ( np.maximum(array, 0) / np.amax(array) ) * 255.0
    array = array.astype(np.uint8)

    # shape = array.shape
    image = Image.fromarray(array)
    image.save(path_file)

class RsnaMiccai(GeneratorBasedBuilder):
    """
    RSNA-MICCAI Dataset.
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
                "image": ImageF(encoding_format = "jpeg"),
                "label": Text()
            }),
            supervised_keys = ("image", "label"),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_extracted = dl_manager.download_kaggle_data(_DATASET_KAGGLE)

        return [
            SplitGenerator(
                name = tfds.Split.TRAIN,
                gen_kwargs = {
                    "path":  path_extracted,
                    "type_": "train",
                }
            )
        ]
        
    def _generate_examples(self, path, type_):
        pass