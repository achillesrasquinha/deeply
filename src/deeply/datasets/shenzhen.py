import os, os.path as osp
from glob import glob

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

from deeply.datasets.montgomery import _DATASET_CITATION
from deeply.util.string import strip, safe_decode
from deeply.util.system import makedirs
from deeply.log import get_logger

logger = get_logger()

_DATASET_URL         = "http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip"
_DATASET_KAGGLE      = "yoctoman/shcxr-lung-mask"
_DATASET_HOMEPAGE    = "https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html"
_DATASET_DESCRIPTION = """
The Shenzhen dataset was collected in collaboration with Shenzhen No.3 People’s Hospital, Guangdong Medical College, Shenzhen, China. The chest X-rays are from outpatient clinics and were captured as part of the daily hospital routine within a 1-month period, mostly in September 2012, using a Philips DR Digital Diagnost system. The set contains 662 frontal chest X-rays, of which 326 are normal cases and 336 are cases with manifestations of TB, including pediatric X-rays (AP). The X-rays are provided in PNG format. Their size can vary but is approximately 3K × 3K pixels.
"""

def img_mask_open(img):
    i = Image.open(img)
    i = i.convert("L")
    return i

class Shenzhen(GeneratorBasedBuilder):
    """
    Shenzhen Hospital Chest X-ray Dataset.
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
                #   "sex": Text(),
                #   "age": Text(),
                # "label": Text(),
            }),
            supervised_keys = ("image", "mask"),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_extracted_images = dl_manager.download_and_extract(_DATASET_URL)
        path_masks = dl_manager.download_kaggle_data(_DATASET_KAGGLE)

        return {
            "data": self._generate_examples(
                images_path = osp.join(path_extracted_images, "ChinaSet_AllFiles"),
                masks_path  = osp.join(path_masks, "mask")
            )
        }
        
    def _generate_examples(self, images_path, masks_path):
        path_images = osp.join(images_path, "CXR_png")
        path_data   = osp.join(images_path, "ClinicalReadings")

        for path_img in glob(osp.join(path_images, "*.png")):
            fname  = osp.basename(osp.normpath(path_img))
            prefix = str(fname).split(".png")[0]

            path_mask = osp.join(masks_path, "%s_mask.png" % prefix)

            if not osp.exists(path_mask):
                logger.warn("Unable to find mask for image: %s" % prefix)
                continue

            path_txt = osp.join(path_data, "%s.txt" % prefix)

            with open(path_txt) as f:
                content = f.readlines()
                lines   = list(filter(bool, [strip(line) for line in content]))

                print(lines)

                # sex       = list(map(lambda x: safe_decode(strip(x)), lines[0].split(" ")))
                # age       = "".join((i for i in age if i.isdigit()))

                # if len(lines) != 1:
                #     label = safe_decode(strip(lines[1]))
                # else:
                #     label = ""

                yield prefix, {
                    "image": path_img,
                     "mask": path_mask,
                    #   "sex": sex,
                    #   "age": age,
                    # "label": label
                }