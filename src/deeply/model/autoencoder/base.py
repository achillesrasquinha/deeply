from tensorflow.keras.optimizers import Adam

from deeply.model.base import BaseModel
from deeply.const import DEFAULT

class AutoEncoder(BaseModel):
    def __init__(self, encoder, decoder, *args, **kwargs):
        kwargs["inputs"]  = []
        kwargs["outputs"] = []
        
        super_ = super(AutoEncoder, self)
        super_.__init__(*args, **kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.loss_fn = {}
        self.optimizers = {}
    
    def compile(self, *args, **kwargs):
        encoder_learning_rate = kwargs.pop("encoder_learning_rate", DEFAULT["generative_model_encoder_learning_rate"])
        decoder_learning_rate = kwargs.pop("decoder_learning_rate", DEFAULT["generative_model_decoder_learning_rate"])

        self.loss_fn["encoder"] = kwargs.pop("encoder_loss")
        self.loss_fn["decoder"] = kwargs.pop("decoder_loss")

        self.optimizers["encoder"] = Adam(learning_rate = encoder_learning_rate)
        self.optimizers["decoder"] = Adam(learning_rate = decoder_learning_rate)

        super_ = super(AutoEncoder, self)
        return super_.compile(*args, **kwargs)