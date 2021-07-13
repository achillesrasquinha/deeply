import numpy as np
from tensorflow.keras.utils import Sequence

from deeply.const import DEFAULT

class BaseDataGenerator(Sequence):
    def __init__(self,
        X = None,
        batch_size = DEFAULT["batch_size"],
        shuffle = False
    ):
        self.batch_size  = batch_size
        self._n_samples  = len(X or [])
        self._shuffle    = shuffle

    @property
    def n_samples(self):
        return getattr(self, "_n_samples", 0)

    def __len__(self):
        return int(np.floor(self.n_samples) / self.batch_size)