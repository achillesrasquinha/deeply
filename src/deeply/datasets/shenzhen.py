from tensorflow_datasets.core import (
    Version,
    GeneratorBasedBuilder,
    DatasetInfo
)
from tensorflow_datasets.core.features import (
    FeaturesDict,
    Image
)

_DATASET_URL = "http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip"
_DATASET_HOMEPAGE = """

"""


class Shezhen(GeneratorBasedBuilder):
    """
    Shezhen Dataset.
    """

    VERSION = Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial Release."
    }

    def _info(self):
        return DatasetInfo(
            builder  = self,
            features = FeaturesDict({
                
            })
        )

    def _split_generators(self, dl_manager):
        path_extracted = dl_manager.download_and_extract(_DATASET_URL)

    def _generate_examples(self, path):
        pass