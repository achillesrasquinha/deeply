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

_DATASET_KAGGLE      = "siim-covid19-detection"
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

class SiimCovid19(GeneratorBasedBuilder):
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
        csv_path   = osp.join(path, "train_image_level.csv")

        dir_images = osp.join(path, "images", type_)
        makedirs(dir_images, exist_ok = True)

        with GFile(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                study_instance_uid = row["StudyInstanceUID"]
                image_uid  = row["id"].split("_image")[0]

                prefix = "%s_%s" % (study_instance_uid, image_uid)
                # path_image = osp.join( dir_images, "%s.jpg" % prefix )

                if not osp.exists(path_image):
                    path_dicom_dir = osp.join( path, type_, study_instance_uid )
                    path_hashes = os.listdir(path_dicom_dir)

                    for i, path_hash in enumerate(path_hashes):
                        path_dicom = osp.join( path_dicom_dir, path_hash, "%s.dcm" % image_uid )
                        path_image = osp.join( dir_images, "%s_%s.jpg" % (prefix, i) )
                    
                        dicom = pydicom.dcmread(path_dicom)
                        dicom_to_image_file(dicom, path_image)

                yield prefix, {
                    "image": path_image,
                    "label": ""
                }