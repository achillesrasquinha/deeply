import sys

from tensorflow.keras.callbacks import Callback

class ProgressStepCallback(Callback):
    def _write(self, str):
        print(str)

    def on_train_begin(self, logs = None):
        # self._write("Begin Training...")
        pass

    def on_train_batch_begin(self, batch, logs = None):
        # self._write("Begin Training Batch...")
        pass
