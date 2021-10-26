import os.path as osp
from glob import glob

from tensorflow_datasets.core import (
    Version,
    GeneratorBasedBuilder,
    DatasetInfo
)
from tensorflow_datasets.core.features import (
    FeaturesDict,
    Image as ImageF
)
from bpyutils.util.system import read

_DATASET_HOMEPAGE    = ""
_DATASET_KAGGLE      = "achillesrasquinha/etislarib"
_DATASET_DESCRIPTION = """
"""
_DATASET_CITATION    = """\
"""
class ETISLarib(GeneratorBasedBuilder):
    """
    The ETIS-Larib Dataset.
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
                 "mask": ImageF(encoding_format = "jpeg")
            }),
            supervised_keys = ("image", "mask"),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_extracted = dl_manager.download_kaggle_data(_DATASET_KAGGLE)
        return {
            "data": self._generate_examples(path = osp.join(path_extracted, "ETIS-Larib"))
        }
        
    def _generate_examples(self, path):
        path_images = osp.join(path, "images")
        path_masks  = osp.join(path, "masks")

        for path_img in glob(osp.join(path_images, "*.jpg")):
            fname  = osp.basename(osp.normpath(path_img))
            prefix = str(fname).split(".jpg")[0]

            path_mask = osp.join(path_masks, fname)

            if osp.exists(path_mask):
                yield prefix, {
                    "image": path_img,
                    "mask": path_mask
                }