import os, os.path as osp
import glob

import tensorflow as tf
from tensorflow_datasets.core import (
    Version,
    GeneratorBasedBuilder,
    DatasetInfo
)
from tensorflow_datasets.core.features import (
    FeaturesDict,
    Image as ImageF,
    Text,
    Tensor,
    # ClassLabel,
)
from PIL import Image

from deeply.util.string import strip, safe_decode
from deeply.util.system import makedirs
from deeply.log import get_logger

logger = get_logger()

_DATASET_URLS        = [
    "http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip",
    "https://drive.google.com/uc?export=download&id=1KN3y7g3OiMsEy-JHqJcIqgpNla7PxnAt"
]
_DATASET_HOMEPAGE    = "https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html"
_DATASET_DESCRIPTION = """

"""
_DATASET_CITATION    = """
"""

def img_mask_open(img):
    i = Image.open(img)
    i = i.convert("L")
    return i

class Shenzhen(GeneratorBasedBuilder):
    """
    Shenzhen Dataset.
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
                "image": ImageF(),
                 "mask": ImageF(),
                  "sex": Text(),
                  "age": Text(),
                "label": Text(),
            }),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_extracted_images, path_extracted_masks = dl_manager.download_and_extract(_DATASET_URLS)
        return {
            "train": self._generate_examples(
                images_path = osp.join(path_extracted_images, "ChinaSet_AllFiles"),
                masks_path  = osp.join(path_extracted_masks,  "mask")
            )
        }
        
    def _generate_examples(self, images_path, masks_path):
        path_images = osp.join(images_path, "CXR_png")
        path_data   = osp.join(images_path, "ClinicalReadings")

        for path_img in os.listdir(path_images):
            prefix   = str(path_img).split(".png")[0]
            path_img = osp.join(path_images, path_img)

            path_mask = osp.join(masks_path, "%s.png" % prefix)
            print(path_mask)

            if not osp.exists(path_mask):
                logger.warn("Unable to find mask for image: %s" % prefix)
                continue

            path_txt  = osp.join(path_data, "%s.txt" % prefix)

            with open(path_txt) as f:
                content = f.readlines()
                lines   = list(filter(bool, [strip(line) for line in content]))

                sex, age  = list(map(lambda x: safe_decode(strip(x)), lines[0].split(" ")))
                age       = int("".join((i for i in age if i.isdigit())))

                if len(lines) != 1:
                    label = safe_decode(strip(lines[1]))
                else:
                    label = ""

                yield prefix, {
                    "image": path_img,
                     "mask": path_mask,
                      "sex": sex,
                      "age": age,
                    "label": label
                }