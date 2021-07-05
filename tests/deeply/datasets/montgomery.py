from tensorflow_datasets.testing import (
    DatasetBuilderTestCase,
    test_main as main
)

from deeply.datasets.montgomery import Montgomery

class MontgomeryTest(DatasetBuilderTestCase):
    DATASET_CLASS = Montgomery

if __name__ == "__main__":
    main()