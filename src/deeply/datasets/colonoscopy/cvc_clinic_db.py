from deeply.datasets.util import image_mask

from tensorflow_datasets.core import (
    Version,
    GeneratorBasedBuilder
)

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
        "1.0.0": "Initial Release"
    }

    def _info(self, *args, **kwargs):
        return image_mask._info(self,
            description = _DATASET_DESCRIPTION,
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION,
            *args, **kwargs
        )

    def _split_generators(self, *args, **kwargs):
        return image_mask._split_generators(self, kaggle = _DATASET_KAGGLE, *args, **kwargs)

    def _generate_examples(self, *args, **kwargs):
        return image_mask._generate_examples(self, *args, **kwargs)