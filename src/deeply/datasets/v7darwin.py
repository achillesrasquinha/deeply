import os, os.path as osp
import json
from   glob import glob

import requests as req
from   PIL import Image, ImageDraw
import imageio

from tensorflow_datasets.core import (
    Version,
    GeneratorBasedBuilder,
    DatasetInfo
)
from tensorflow_datasets.core.features import (
    FeaturesDict,
    Image as ImageF
)
from tqdm import tqdm
from deeply.util.system import makedirs
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

    shape = img_arr.shape
    h, w  = shape[0], shape[1]
    mask  = Image.new("L", (w, h), 0)

    drawer = ImageDraw.Draw(mask)

    for annotation in data["annotations"]:
        if "polygon" in annotation:
            polygon = annotation["polygon"]["path"]
            polygon = [(p["x"], p["y"]) for p in polygon]
            drawer.polygon(polygon, outline = 0, fill = 255)

    mask.save(path_mask)

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
                 "mask": ImageF()
            }),
            supervised_keys = ("image", "mask"),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_extracted = dl_manager.download_and_extract(_DATASET_URL)

        return {
            "data": self._generate_examples(
                path = path_extracted
            )
        }
        
    def _generate_examples(self, path):
        path_images = osp.join(path, "images")
        path_masks  = osp.join(path, "masks")

        makedirs(path_images, exist_ok = True)
        makedirs(path_masks,  exist_ok = True)

        logger.info("Processing images and masks.")

        for path_json in tqdm(glob(osp.join(path, "*.json"))):
            with open(path_json) as f:
                data = json.load(f)

            filename   = data["image"]["filename"]

            path_image = osp.join(path_images, filename)

            if not osp.exists(path_image):
                response = req.get(data["image"]["url"], stream = True)
                response.raise_for_status()

                with open(path_image, "wb") as f:
                    for content in response.iter_content(chunk_size = 1024):
                        f.write(content)

            path_mask = osp.join(path_masks, filename)

            if not osp.exists(path_mask):
                build_mask(data, path_image, path_mask)

            yield path_image, {
                "image": path_image,
                 "mask": path_mask
            }