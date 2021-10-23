import os.path as osp
from glob import glob
import json

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
from bpyutils.util.system import read

_DATASET_HOMEPAGE    = "https://datasets.simula.no/kvasir-seg"
_DATASET_URL         = osp.join(_DATASET_HOMEPAGE, "Kvasir-SEG.zip")
_DATASET_DESCRIPTION = """
Pixel-wise image segmentation is a highly demanding task in medical image analysis. It is difficult to find annotated medical images with corresponding segmentation mask. Here, we present Kvasir-SEG. It is an open-access dataset of gastrointestinal polyp images and corresponding segmentation masks, manually annotated and verified by an experienced gastroenterologist. This work will be valuable for researchers to reproduce results and compare their methods in the future. By adding segmentation masks to the Kvasir dataset, which until today only consisted of framewise annotations, we enable multimedia and computer vision researchers to contribute in the field of polyp segmentation and automatic analysis of colonoscopy videos.
"""
_DATASET_CITATION    = """\
@inproceedings{jha2020kvasir,
    title           = {Kvasir-seg: A segmented polyp dataset},
    author          = {Jha, Debesh and Smedsrud, Pia H and Riegler, Michael A and Halvorsen, P{\aa}l and de Lange, Thomas and Johansen, Dag and Johansen, H{\aa}vard D},
    booktitle       = {International Conference on Multimedia Modeling},
    pages           = {451--462},
    year            = {2020},
    organization    = {Springer}
}
"""
class KvasirSegmented(GeneratorBasedBuilder):
    """
    The Kvasir-SEG Dataset.
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
                 "mask": ImageF(encoding_format = "jpeg"),
                "bounding_box": Text(),
            }),
            supervised_keys = ("image", "mask"),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_extracted = dl_manager.download_and_extract(_DATASET_URL)
        return {
            "data": self._generate_examples(path = osp.join(path_extracted, "Kvasir-SEG"))
        }
        
    def _generate_examples(self, path):
        path_images = osp.join(path, "images")
        path_masks  = osp.join(path, "masks")
        path_bboxes = osp.join(path, "kavsir_bboxes.json")

        bboxes      = json.loads(read(path_bboxes))

        for path_img in glob(osp.join(path_images, "*.jpg")):
            fname  = osp.basename(osp.normpath(path_img))
            prefix = str(fname).split(".jpg")[0]

            path_mask = osp.join(path_masks, fname)

            yield prefix, {
                "image": path_img,
                 "mask": path_mask,
                "bounding_box": str(bboxes[prefix])
            }