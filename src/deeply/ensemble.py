import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import (
    # Layer,
    Concatenate
)

from deeply.model.base     import BaseModel
from deeply.model.logistic import LogisticRegression

class StackingModel(BaseModel):
    def __init__(self, models = None, *args, **kwargs):
        self._super  = super(StackingModel, self)
        self._super.__init__(*args, **kwargs)

        self._models = models

    def _stack_input(self, data):
        X, y   = data
        length = len(self._models)

        X = [X] * length
        
        return X, y

    def train_step(self, data):
        X, y = self._stack_input(data)
        
        with tf.GradientTape() as tape:
            y_pred = self(X, training = True)
            loss   = self.compiled_loss(y, y_pred,
                regularization_losses = self.losses
            )

        trainable_vars = self.trainable_variables
        gradients      = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)

        return { m.name: m.result() for m in self.metrics }

    def test_step(self, data):
        X, y   = self._stack_input(data)
        y_pred = self(X, training = False)

        self.compiled_loss(y, y_pred,
            regularization_losses = self.losses
        )

        self.compiled_metrics.update_state(y, y_pred)

        return { m.name: m.result() for m in self.metrics }

def Stacking(models,
    final_model = LogisticRegression,
    name        = "stacking"
):
    """
        Stacking Ensemble.

        :param final_model: Final Model for meta-learning.
    """
    # rename model layers
    for model in models:
        for layer in model.layers:
            layer._name = "%s_%s_%s" % (name, model.name, layer.name)

    inputs, outputs = list(zip(*[( model.input, model.output ) for model in models]))
    
    output = Concatenate()(outputs)
    
    model  = StackingModel(inputs = inputs, outputs = output, name = name,
        models = models)

    return model