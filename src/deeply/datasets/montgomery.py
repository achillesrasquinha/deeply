import os, os.path as osp

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

_DATASET_URL         = "http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
_DATASET_HOMEPAGE    = "https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html"
_DATASET_DESCRIPTION = """

"""
_DATASET_CITATION    = """

"""

def img_mask_open(img):
    i = Image.open(img)
    i = i.convert("L")
    return i

class Montgomery(GeneratorBasedBuilder):
    """
    Montgomery Dataset.
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
        path_extracted = dl_manager.download_and_extract(_DATASET_URL)
        return {
            "train": self._generate_examples(path = osp.join(path_extracted, "MontgomerySet"))
        }
        
    def _generate_examples(self, path):
        path_images = osp.join(path, "CXR_png")
        path_data   = osp.join(path, "ClinicalReadings")
        path_masks  = osp.join(path, "ManualMask")
        path_masks_merged = osp.join(path_masks, "merged")

        makedirs(path_masks_merged, exist_ok = True)

        for path_img in os.listdir(path_images):
            prefix   = str(path_img).split(".png")[0]
            path_img = osp.join(path_images, path_img)

            path_txt  = osp.join(path_data, "%s.txt" % prefix)

            path_mask = osp.join(path_masks_merged, "%s.png" % prefix)

            if not osp.exists(path_mask):
                path_mask_left  = osp.join(path_masks, "leftMask",  "%s.png" % prefix)
                path_mask_right = osp.join(path_masks, "rightMask", "%s.png" % prefix)

                img_mask_left   = img_mask_open(path_mask_left)
                img_mask_right  = img_mask_open(path_mask_right)

                img_mask = Image.blend(img_mask_left, img_mask_right, 0.5)
                img_mask = img_mask.convert("1")
                img_mask.save(path_mask)

            with open(path_txt) as f:
                content = f.readlines()
                lines   = list(filter(bool, [strip(line) for line in content]))

                sex     = safe_decode(strip(lines[0].split(": ")[1]))
                age     = safe_decode(strip(lines[1].split(": ")[1].split("Y")[0]))
                label   = safe_decode(strip(lines[2]))

                yield prefix, {
                    "image": path_img,
                     "mask": path_mask,
                      "sex": sex,
                      "age": age,
                    "label": label
                }