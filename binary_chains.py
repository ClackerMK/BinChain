import typing as tp

from chainer import types, variable, Chain, ChainList

from chainer import functions as F
from chainer import links as L

import binary_links as BL
import binarization_functions as BF


# Macro Layers
class BNNDenseChain(Chain):

    def __init__(self,
                 size_in: tp.Optional[int],
                 size_out: int,
                 binarization_fn: tp.Callable[[variable.Variable], variable.Variable]
                 = BF.deterministic_binarization,
                 activation_fn: tp.Optional[tp.Callable[[variable.Variable], variable.Variable]]
                 = BF.deterministic_binarization,
                 prelu: bool = True,
                 gain: bool = True,
                 dropout_p: float = 0.,
                 residual_levels: int = 0,
                 ):
        super(BNNDenseChain, self).__init__()

        self.size_in = size_in
        self.size_out = size_out
        self.prelu = prelu
        self.dropout_p = dropout_p

        with self.init_scope():
            self.batch_norm_l = L.BatchNormalization(size_in)
            self.linear_l = BL.BinaryLinear(in_size=size_in, out_size=size_out,
                                            binarization_fn=binarization_fn, gain=gain and not prelu)

            if residual_levels > 0:
                self.activation_fn = BL.ResidualBinarization(residual_levels)
            elif gain and activation_fn is not None:
                self.scale_l = L.Scale(W_shape=(1,))
                self.activation_fn = lambda x: self.scale_l(activation_fn(x))
            else:
                self.activation_fn = activation_fn

            if prelu:
                self.prelu_l = L.PReLU((size_out, ))
            else:
                self.prelu_l = None


    def forward(self, x):
        if len(x.shape) > 2:
            x = F.reshape(x, (x.shape[0], -1))

        if self.dropout_p > 0.:
            x = F.dropout(x, self.dropout_p)

        x = self.batch_norm_l(x)

        if self.activation_fn is not None:
            x = self.activation_fn(x)

        x = self.linear_l(x)

        if self.prelu:
            x = self.prelu_l(x)
        return x

    @classmethod
    def from_params(cls, *args, **kwargs):
        raise NotImplementedError


class BNNConvChain(Chain):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ksize: tp.Union[int, tp.Tuple[int, int]],
                 stride: int = 1,
                 pad: int = 0,
                 initialW: tp.Optional[types.InitializerSpec]
                 = None,
                 binarization_fn: tp.Callable[[variable.Variable], variable.Variable]
                 = BF.deterministic_binarization,
                 activation_fn: tp.Optional[tp.Callable[[variable.Variable], variable.Variable]]
                 = BF.deterministic_binarization,
                 prelu: bool = True,
                 prelu_shared: bool = True,
                 pooling: tp.Optional[tp.Callable] = F.max_pooling_2d,
                 pooling_size: tp.Union[int, tp.Tuple[int, int]] = 2,
                 pooling_stride: tp.Optional[tp.Union[int, tp.Tuple[int, int]]] = None,
                 dropout_p: float = 0.,
                 gain: bool = True,
                 residual_levels: int = 0,
                 ):
        super(BNNConvChain, self).__init__()
        self.prelu = prelu
        self.dropout_p = dropout_p
        self.pooling = pooling
        if pooling is not None:
            self.pooling_size = pooling_size
            self.pooling_stride = pooling_stride
            self.pooling_pad = 0

        with self.init_scope():
            self.batch_norm_l = L.BatchNormalization(in_channels)
            self.conv_l = BL.BinaryConvolution2D(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 ksize=ksize,
                                                 stride=stride,
                                                 pad=pad,
                                                 initialW=initialW,
                                                 binarization_fn=binarization_fn,
                                                 gain=gain and not prelu)

            if residual_levels > 0:
                self.activation_fn = BL.ResidualBinarization(residual_levels)
            elif gain and activation_fn is not None:
                self.scale_l = L.Scale(W_shape=(1,))
                self.activation_fn = lambda x: self.scale_l(activation_fn(x))
            else:
                self.activation_fn = activation_fn

            if self.prelu:
                self.prelu_l = L.PReLU((out_channels, ) if prelu_shared else (1, ))
            else:
                self.prelu_l = None

    def forward(self, x):
        if self.dropout_p > 0.:
            x = F.dropout(x, self.dropout_p)

        x = self.batch_norm_l(x)

        if self.activation_fn is not None:
            x = self.activation_fn(x)

        x = self.conv_l(x)

        if self.prelu:
            x = self.prelu_l(x)

        if self.pooling is not None:
            x = self.pooling(x, ksize=self.pooling_size, stride=self.pooling_stride, pad=self.pooling_pad)

        return x

    @classmethod
    def from_params(cls, *args, **kwargs):
        raise NotImplementedError


# noinspection PyAbstractClass
class CouvBNN(ChainList):
    def __init__(self, nogain=False, noprelu=False, residual_levels=0):
        gain_kwargs = {
            'gain': not nogain,
            'prelu': not noprelu,
            'residual_levels': residual_levels
        }
        super(CouvBNN, self).__init__(
            BNNDenseChain(784, 4096, dropout_p=.2, activation_fn=None, **gain_kwargs),
            BNNDenseChain(4096, 4096, dropout_p=.5, **gain_kwargs),
            BNNDenseChain(4096, 4096, dropout_p=.5, **gain_kwargs),
            BNNDenseChain(4096, 10, dropout_p=.5, **gain_kwargs),
            L.Scale(W_shape=(10,))
        )

    def forward(self, x):
        for f in self.children():
            x = f(x)
        return x


# noinspection PyAbstractClass
class BinLeNet(ChainList):
    def __init__(self, nogain=False, noprelu=False, residual_levels=0):
        gain_kwargs = {
            'gain': not nogain,
            'prelu': not noprelu,
            'residual_levels': residual_levels
        }

        super(BinLeNet, self).__init__(BNNConvChain(in_channels=1, out_channels=6, ksize=5, stride=1,
                                                    pooling_size=2, activation_fn=None,
                                                    **gain_kwargs),
                                       BNNConvChain(in_channels=6, out_channels=16, ksize=5, stride=1,
                                                    pooling_size=2, **gain_kwargs),
                                       BNNConvChain(in_channels=16, out_channels=120, ksize=4, stride=1,
                                                    pooling=None, **gain_kwargs),
                                       BNNDenseChain(120, 84, **gain_kwargs),
                                       BNNDenseChain(84, 10, **gain_kwargs),
                                       L.Scale(W_shape=(10,))
                                       )

    def forward(self, x):
        for f in self.children():
            x = f(x)
        return x


# noinspection PyAbstractClass
class BinCifar10ConvNet(ChainList):

    def __init__(self, nogain=False, noprelu=False, residual_levels=0):
        gain_kwargs = {
            'gain': not nogain,
            'prelu': not noprelu,
            'residual_levels': residual_levels
        }

        super(BinCifar10ConvNet, self).__init__(BNNConvChain(in_channels=3, out_channels=32,
                                                             ksize=3, pad=1,
                                                             pooling=None,
                                                             **gain_kwargs),
                                                BNNConvChain(in_channels=32, out_channels=32,
                                                             ksize=3, pad=1,
                                                             pooling_size=2,
                                                             **gain_kwargs),
                                                BNNConvChain(in_channels=32, out_channels=64,
                                                             ksize=3, pad=1,
                                                             pooling=None,
                                                             dropout_p=.2,
                                                             **gain_kwargs),
                                                BNNConvChain(in_channels=64, out_channels=64,
                                                             ksize=3, pad=1,
                                                             pooling_size=2,
                                                             **gain_kwargs),
                                                BNNConvChain(in_channels=64, out_channels=128,
                                                             ksize=3, pad=1,
                                                             pooling=None,
                                                             dropout_p=.3,
                                                             **gain_kwargs),
                                                BNNConvChain(in_channels=128, out_channels=128,
                                                             ksize=3, pad=1,
                                                             pooling_size=2,
                                                             **gain_kwargs),
                                                BNNDenseChain(size_in=2048, size_out=10,
                                                              dropout_p=.4,
                                                              **gain_kwargs))

    def forward(self, x):
        for f in self.children():
            x = f(x)
        return x

