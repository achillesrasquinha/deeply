import os.path as osp

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

_DATASET_HOMEPAGE    = "https://datasets.simula.no/hyper-kvasir"
_DATASET_URL         = osp.join(_DATASET_HOMEPAGE, "hyper-kvasir.zip")
_DATASET_DESCRIPTION = """
The data is collected during real gastro- and colonoscopy examinations at a Hospital in Norway and partly labeled by experienced gastrointestinal endoscopists. The dataset contains 110,079 images and 374 videos where it captures anatomical landmarks and pathological and normal findings. Resulting in around 1 million images and video frames all together.
"""
_DATASET_CITATION    = """\
@misc{borgli2020,
    title={Hyper-Kvasir: A Comprehensive Multi-Class Image and Video Dataset for Gastrointestinal Endoscopy},
    url={osf.io/mkzcq},
    DOI={10.31219/osf.io/mkzcq},
    publisher={OSF Preprints},
    author={Borgli, Hanna and Thambawita, Vajira and Smedsrud, Pia H and Hicks, Steven and Jha, Debesh and Eskeland, Sigrun L and Randel, Kristin R and Pogorelov, Konstantin and Lux, Mathias and Nguyen, Duc T D and Johansen, Dag and Griwodz, Carsten and Stensland, H{\aa}kon K and Garcia-Ceja, Enrique and Schmidt, Peter T and Hammer, Hugo L and Riegler, Michael A and Halvorsen, P{\aa}l and de Lange, Thomas},
    year={2019},
    month={Dec}
}
"""

class HyperKvasir(GeneratorBasedBuilder):
    """
    HyperKvasir Dataset.
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
                "label": Text(),
            }),
            supervised_keys = ("image", "label"),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_extracted = dl_manager.download_and_extract(_DATASET_URL)
        
    def _generate_examples(self, path):
        pass