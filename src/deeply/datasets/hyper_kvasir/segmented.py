import os.path as osp
import json

from tensorflow.io.gfile import (
    GFile
)
from tensorflow_datasets.core import (
    Version,
    GeneratorBasedBuilder,
    DatasetInfo
)
from tensorflow_datasets.core.features import (
    FeaturesDict,
    Image as ImageF,
    Text
)
from bpyutils._compat import iteritems
from deeply.datasets.hyper_kvasir.base import (
    _DATASET_HOMEPAGE,
    _DATASET_DESCRIPTION,
    _DATASET_CITATION
)

_DATASET_URL = osp.join(_DATASET_HOMEPAGE, "hyper-kvasir-segmented-images.zip")

class HyperKvasirSegmented(GeneratorBasedBuilder):
    """
    HyperKvasir Segmented Dataset.
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
                "mask":  ImageF(encoding_format = "jpeg"),
                "bounding_box": Text()
            }),
            supervised_keys = ("image", "mask"),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_extracted = dl_manager.download_and_extract(_DATASET_URL)
        return {
            "data": self._generate_examples(path = osp.join(path_extracted, "segmented-images"))
        }

    def _generate_examples(self, path):
        json_path = osp.join(path, "bounding-boxes.json")
        
        with GFile(json_path) as f:
            data = json.load(f)

            for name, box in iteritems(data):
                filename   = "%s.jpg" % name
                
                path_image = osp.join(path, "images", filename)
                path_mask  = osp.join(path, "masks",  filename)

                yield name, {
                    "image": path_image,
                    "mask":  path_mask,
                    "bounding_box": str(box)
                }