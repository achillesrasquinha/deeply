import os.path as osp

from tensorflow_datasets.core import (
    Version,
    GeneratorBasedBuilder,
    DatasetInfo
)
from tensorflow_datasets.core.features import (
    FeaturesDict,
    Image
)

_DATASET_URL         = "https://github.com/v7labs/covid-19-xray-dataset/blob/master/annotations/all-images.zip?raw=true"
_DATASET_DESCRIPTION = """

"""
_DATASET_CITATION = """

"""

class V7Darwin(GeneratorBasedBuilder):
    """
    V7 Darwin Dataset.
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
                "image": Image(),
            }),
            homepage    = "",
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_extracted = dl_manager.download_and_extract(_DATASET_URL)
        return {
            "images": self._generate_examples(path = path_extracted / "all-images")
        }
        
    def _generate_examples(self, path):
        for path_img in path.glob("*.json"):
            pass