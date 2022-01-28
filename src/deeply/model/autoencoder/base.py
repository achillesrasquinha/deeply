from deeply.model.base import BaseModel

class AutoEncoder(BaseModel):
    def __init__(self, encoder, decoder, *args, **kwargs):
        kwargs["inputs"]  = []
        kwargs["outputs"] = []

        self._super  = super(AutoEncoder, self)
        self._super.__init__(*args, **kwargs)

        self.encoder = encoder
        self.decoder = decoder