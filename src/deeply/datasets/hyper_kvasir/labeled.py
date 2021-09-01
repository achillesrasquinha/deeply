import os.path as osp
import csv

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
from deeply.datasets.hyper_kvasir.base import (
    _DATASET_HOMEPAGE,
    _DATASET_DESCRIPTION,
    _DATASET_CITATION
)

_DATASET_URL = osp.join(_DATASET_HOMEPAGE, "hyper-kvasir-labeled-images.zip")

class HyperKvasirLabeled(GeneratorBasedBuilder):
    """
    HyperKvasir Labeled Dataset.
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
                "organ": Text(),
                "finding": Text(),
                "label": Text(),
            }),
            supervised_keys = ("image", "label"),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_extracted = dl_manager.download_and_extract(_DATASET_URL)
        return {
            "data": self._generate_examples(path = path_extracted)
        }

    def _generate_examples(self, path):
        csv_path = osp.join(path, "image-labels.csv")
        
        with GFile(csv_path) as f:
            reader = csv.DictReader(f)

            for row in reader:
                image_hash  = row["Video file"]
                image_fname = "%s.jpg" % row["Video file"]
                organ       = row["Organ"]
                finding     = row["Finding"]
                label       = row["Classification"]

                path_organ = "lower-gi-tract" if organ == "Lower GI" else "upper-gi-tract"

                path_image = osp.join(path, path_organ, label, finding, image_fname)

                yield image_hash, {
                    "image": path_image,
                    "organ": organ,
                    "finding": finding,
                    "label": label
                }