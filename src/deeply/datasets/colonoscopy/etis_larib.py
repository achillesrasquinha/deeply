from deeply.datasets.util import image_mask

from tensorflow_datasets.core import (
    Version,
    GeneratorBasedBuilder
)

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
        return image_mask._generate_examples(*args, **kwargs)