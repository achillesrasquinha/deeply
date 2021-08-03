import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import (
    # Layer,
    Concatenate
)
from tensorflow.keras.callbacks import Callback

from tqdm.keras import TqdmCallback

import tqdm as tq

from deeply.model.base     import BaseModel
from deeply.model.logistic import LogisticRegression
from deeply.util._dict     import merge_dict

def get_model_loss(model, data, training = False):
    X, y   = data

    y_pred = model(X, training = training)

    loss   = model.compiled_loss(y, y_pred,
        regularization_losses = model.losses
    )

    return model, y_pred, loss

def _concat(x, y):
    return tf.concat((x, y), 0) if x is not None else y
class StackingModelCallback(Callback):
    def __init__(self, fit_args, mapper = None, epochs = 1, *args, **kwargs):
        self._super = super(StackingModelCallback, self)
        self._super.__init__(*args, **kwargs)

        self._fit_args = fit_args
        self._mapper   = mapper

        self._epochs   = epochs

    def on_train_batch_end(self, batch, logs = None):
        model   = self.model

        models  = model.models
        meta_learner = model.meta_learner

        data    = self._fit_args["args"][0]
        fit_kwargs = self._fit_args.get("kwargs", { })

        verbose = fit_kwargs.get("verbose", 0)
        X_meta, y_meta = None, None

        mapper  = self._mapper

        epochs  = self._epochs

        for m in models:
            for X, y in data:
                y_pred = m.predict(X)

                if mapper:
                    y_pred, y = mapper(y_pred, y)

                X_meta = _concat(X_meta, y_pred)
                y_meta = _concat(y_meta, y)

        meta_learner.fit(X_meta, y_meta, epochs = epochs, verbose = verbose)
class StackingModel(BaseModel):
    def __init__(self, models = None, final_model = None, *args, **kwargs):
        self._super  = super(StackingModel, self)
        self._super.__init__(*args, **kwargs)

        self._models      = models
        self._final_model = final_model

    @property
    def models(self):
        return getattr(self, "_models", [])

    @property
    def meta_learner(self):
        return getattr(self, "_final_model", LogisticRegression())

    def compile(self, *args, **kwargs):
        for i, _ in enumerate(self._models):
            self._models[i].compile(*args, **kwargs)
        self._super.compile()

    def train_step(self, data):
        X, y = data

        for i, model in enumerate(self.models):
            with tf.GradientTape() as tape:
                y_pred, loss = get_model_loss(model, data, training = True)

            trainable_vars = model.trainable_variables
            gradients      = tape.gradient(loss, trainable_vars)

            model.optimizer.apply_gradients(zip(gradients, trainable_vars))

            model.compiled_metrics.update_state(y, y_pred)

            self._models[i] = model

        return self._get_models_metrics()

    def test_step(self, data):
        X, y = data

        for i, model in enumerate(self.models):
            model, y_pred, _ = get_model_loss(model, data, training = False)

            model.compiled_metrics.update_state(y, y_pred)

            self._models[i] = model

        return self._get_models_metrics()

    def _get_models_metrics(self):
        metrics = { }

        for model in self.models:
            metrics = merge_dict(metrics, { "%s-%s" % (model.name, m.name): m.result() for m in model.metrics })
        
        return metrics

    def fit(self, *args, **kwargs):
        meta_mapper = kwargs.pop("meta_mapper", None)
        meta_epochs = kwargs.pop("meta_epochs", 1)

        callbacks   = kwargs.pop("callbacks", [])
        callbacks.append(StackingModelCallback(
            fit_args = { "args": args, "kwargs": kwargs },
            mapper   = meta_mapper,
            epochs   = meta_epochs
        ))

        kwargs["callbacks"] = callbacks

        return self._super.fit(*args, **kwargs)

def Stacking(models,
    final_model = LogisticRegression(),
    name = "stacking"
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
        models = models, final_model = final_model)

    return model