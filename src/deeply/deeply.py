from __future__ import absolute_import

from deeply.model.factory import ModelFactory
from deeply.const import DEFAULT

def hub(name, *args, **kwargs):
    pretrained = kwargs.pop("pretrained", True)
    if pretrained:
        kwargs["include_top"] = False
        kwargs["weights"]     = kwargs.get("weights", DEFAULT["weights"])

    model = ModelFactory.get(name, *args, **kwargs)
    return model