import math
from collections import OrderedDict

import torch
from torch.nn import *


class PositionalEncoding(Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Classifier(Module):
    def __init__(self):
        super(Classifier, self).__init__()

    def convolutionalMotionEncoder(self, x):
        x = ConvTranspose1d(247, 384, kernel_size=(4,), stride=(2,), padding=(1,))(x)
        x = Dropout(p=0.2, inplace=True)(x)
        x = LeakyReLU(negative_slope=0.2, inplace=True)(x)
        x = Conv1d(384, 512, kernel_size=(4,), stride=(2,), padding=(1,))(x)
        x = Dropout(p=0.2, inplace=True)(x)
        x = LeakyReLU(negative_slope=0.2, inplace=True)(x)
        x = Linear(in_features=512, out_features=512, bias=True)
        return x

    def convolutionalMotionDecoder(self, x):
        x = ConvTranspose1d(512, 384, kernel_size=(4,), stride=(2,), padding=(1,))(x)
        x = LeakyReLU(negative_slope=0.2, inplace=True)(x)
        x = ConvTranspose1d(384, 251, kernel_size=(4,), stride=(2,), padding=(1,))(x)
        x = LeakyReLU(negative_slope=0.2, inplace=True)(x)
        x = Linear(in_features=251, out_features=251, bias=True)
        return x

    def textEncoder(self, x):
        x = Linear(in_features=15, out_features=300, bias=True)(x)
        x = Linear(in_features=300, out_features=512, bias=True)(x)
        x = GRU(512, 512, batch_first=True, bidirectional=True)
        return x

    def priorNetwork(self, x):
        x = Linear(in_features=1024, out_features=1024, bias=True)(x)
        x = Sequential(
            OrderedDict(
                [
                    ("0", Linear(in_features=1024, out_features=1024, bias=True)),
                    ("1", LayerNorm((1024,), eps=1e-05, elementwise_affine=True)),
                    ("2", LeakyReLU(negative_slope=0.2, inplace=True)),
                ]
            )
        )(x)
        x = ModuleList([GRUCell(1024, 1024)])(x)
        x = PositionalEncoding()(x)
        x = Linear(in_features=1024, out_features=128, bias=True)(x)
        x = Linear(in_features=1024, out_features=128, bias=True)(x)
        return x

    def posteriorNetwork(self, x):
        x = Linear(in_features=1024, out_features=1024, bias=True)(x)
        x = Sequential(
            OrderedDict(
                [
                    ("0", Linear(in_features=1536, out_features=1024, bias=True)),
                    ("1", LayerNorm((1024,), eps=1e-05, elementwise_affine=True)),
                    ("2", LeakyReLU(negative_slope=0.2, inplace=True)),
                ]
            )
        )(x)
        x = ModuleList([GRUCell(1024, 1024)])(x)
        x = PositionalEncoding()(x)
        x = Linear(in_features=1024, out_features=128, bias=True)(x)
        x = Linear(in_features=1024, out_features=128, bias=True)(x)
        return x

    def generator(self, x):
        x = Linear(in_features=1024, out_features=1024, bias=True)(x)
        x = Sequential(
            OrderedDict(
                [
                    ("0", Linear(in_features=1152, out_features=1024, bias=True)),
                    ("1", LayerNorm((1024,), eps=1e-05, elementwise_affine=True)),
                    ("2", LeakyReLU(negative_slope=0.2, inplace=True)),
                ]
            )
        )(x)
        x = ModuleList([GRUCell(1024, 1024)])(x)
        x = PositionalEncoding()(x)
        x = Sequential(
            OrderedDict(
                [
                    ("0", Linear(in_features=1024, out_features=1024, bias=True)),
                    ("1", LayerNorm((1024,), eps=1e-05, elementwise_affine=True)),
                    ("2", LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("3", Linear(in_features=1024, out_features=512, bias=True)),
                ]
            )
        )(x)
        return x

    def attentionLayer(self, x):
        x = Linear(in_features=1024, out_features=512, bias=True)(x)
        x = Linear(in_features=1024, out_features=512, bias=False)(x)
        x = Linear(in_features=1024, out_features=512, bias=True)(x)
        x = Softmax(dim=1)(x)
        return x
