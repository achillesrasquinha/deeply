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

_DATASET_HOMEPAGE    = "https://polyp.grand-challenge.org/CVCClinicDB/"
_DATASET_KAGGLE      = "achillesrasquinha/cvcclinicdb"
_DATASET_DESCRIPTION = """
CVC-ClinicDB is a database of frames extracted from colonoscopy videos. These frames contain several examples of polyps. In addition to the frames, we provide the ground truth for the polyps. This ground truth consists of a mask corresponding to the region covered by the polyp in the image
"""
_DATASET_CITATION    = """\
Bernal, J., Sánchez, F. J., Fernández-Esparrach, G., Gil, D., Rodríguez, C., & Vilariño, F. (2015). WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians. Computerized Medical Imaging and Graphics, 43, 99-111
"""
class CVCClinicDB(GeneratorBasedBuilder):
    """
    The CVC-ClinicDB Dataset.
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
            "data": self._generate_examples(path = osp.join(path_extracted, "CVC-ClinicDB"))
        }
        
    def _generate_examples(self, path):
        path_images = osp.join(path, "images")
        path_masks  = osp.join(path, "masks")

        for path_img in glob(osp.join(path_images, "*.jpg")):
            fname  = osp.basename(osp.normpath(path_img))
            prefix = str(fname).split(".jpg")[0]

            path_mask = osp.join(path_masks, fname)

            yield prefix, {
                "image": path_img,
                 "mask": path_mask
            }