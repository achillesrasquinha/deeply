import os, os.path as osp
import json
from   glob import glob

import requests as req

import numpy as np
import tensorflow as tf
from tensorflow_datasets.core import (
    Version,
    GeneratorBasedBuilder,
    DatasetInfo
)
from tensorflow_datasets.core.features import (
    FeaturesDict,
    Image as ImageF,
    Tensor,
    Text
)
from tqdm import tqdm
from deeply.datasets.montgomery import (
    _str_to_int,
    sanitize_lines
)
from deeply.util.system import makedirs
from deeply.util.string import strip, safe_decode
from deeply.log import get_logger

logger = get_logger()

_DATASET_URL         = "https://github.com/v7labs/covid-19-xray-dataset/blob/master/annotations/all-images.zip?raw=true"
_DATASET_HOMEPAGE    = "https://darwin.v7labs.com/v7-labs/covid-19-chest-x-ray-dataset"
_DATASET_DESCRIPTION = """
"""
_DATASET_CITATION    = """

"""

def build_mask(data, path_image, path_mask):
    img_arr  = imageio.imread(path_image)
    mask_arr = np.zeros(shape = img_arr.shape, dtype = np.uint8) 

class V7Darwin(GeneratorBasedBuilder):
    """
    V7Darwin Chest X-ray Dataset.
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
                #   "age": Tensor(shape = (), dtype = tf.uint8),
                # "label": Text(),
            }),
            supervised_keys = ("image", "mask"),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_extracted = dl_manager.download_and_extract(_DATASET_URL)

        return {
            "data": self._generate_examples(
                path = osp.join(path_extracted, "all-images")
            )
        }
        
    def _generate_examples(self, path):
        path_images = osp.join(path, "images")
        path_masks  = osp.join(path, "masks")

        makedirs(path_images, exists_ok = True)
        makedirs(path_masks,  exists_ok = True)

        for path_json in tqdm(glob(osp.join(path, "*.json"))):
            with open(path_json) as f:
                data = json.load(f)

            filename   = data["image"]["filename"]

            path_image = osp.join(path_images, filename)

            if not osp.exists(path_image):
                response = req.get(data["image"]["url"], stream = True)
                response.raise_for_status()

                with open(path_image) as f:
                    for content in response.iter_content(chunk_size = 1024):
                        f.write(content)

            path_mask = osp.join(path_masks, filename)

            if not osp.exists(path_mask):
                build_mask(data, path_image, path_mask)
            
            # fname  = osp.basename(osp.normpath(path_img))
            # prefix = str(fname).split(".png")[0]

            # path_mask = osp.join(masks_path, "%s_mask.png" % prefix)

            # if not osp.exists(path_mask):
            #     logger.warn("Unable to find mask for image: %s" % prefix)
            #     continue

            # path_txt = osp.join(path_data, "%s.txt" % prefix)

            # with open(path_txt) as f:
            #     content = f.readlines()
            #     lines   = sanitize_lines(content)
                
            #     # sex       = list(map(lambda x: safe_decode(strip(x)), lines[0].split(" ")))
            #     # age       = _str_to_int(safe_decode(strip(lines[0].split(" ")[1])))

            #     if len(lines) != 1:
            #         label = safe_decode(strip(lines[1]))
            #     else:
            #         label = ""

            #     yield prefix, {
            #         "image": path_img,
            #          "mask": path_mask,
            #         #   "sex": sex,
            #         #   "age": age,
            #         "label": label
            #     }