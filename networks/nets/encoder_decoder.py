from abc import abstractmethod
from collections import OrderedDict

from .. import BasicNN


class Encoder(BasicNN):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Encoder解码器类是抽象类，需要继承之后才能使用")


class Decoder(BasicNN):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError()

class EncoderDecoder(BasicNN):

    def __init__(self, encoder, decoder, *args, **kwargs):
        super(EncoderDecoder, self).__init__(*args, OrderedDict([
            ('encoder', encoder), ('decoder', decoder)
        ]), **kwargs)

    def forward(self, input):
        enc_X, dec_X = input
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder((dec_X, dec_state))
