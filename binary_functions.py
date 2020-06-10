import numpy as np
from chainer import cuda, FunctionNode, backend, Variable
from chainer import functions as F
from chainer import links as L
from chainer.functions.connection.convolution_2d import Convolution2DFunction
from chainer.functions.connection.linear import LinearFunction
from chainer.utils import type_check
from chainer import utils
from chainer.utils import argument


import binarization_functions


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


# noinspection PyAbstractClass
class BinaryLinear(LinearFunction):

    def __init__(self,
                 binarization_fn=binarization_functions.deterministic_binarization):
        super(BinaryLinear, self).__init__()
        self.binarization_fn = binarization_fn
        self.W_b = None

    def apply(self, inputs):
        # Unpack inputs
        x, W = inputs

        # Binarize weights
        W_b = self.binarization_fn(W)

        # Compute output
        y = super(LinearFunction, self).apply((x, W_b))
        return y


# noinspection PyAbstractClass
class BinaryConvolution2D(Convolution2DFunction):

    def __init__(self,
                 stride=1,
                 pad=0,
                 cover_all=False,
                 binarization_fn=binarization_functions.deterministic_binarization,
                 **kwargs):
        dilate, groups, cudnn_fast = argument.parse_kwargs(
            kwargs, ('dilate', 1), ('groups', 1), ('cudnn_fast', False),
            deterministic='deterministic argument is not supported anymore. '
                          'Use chainer.using_config(\'cudnn_deterministic\', value) '
                          'context where value is either `True` or `False`.')
        super(BinaryConvolution2D, self).__init__(stride=stride,
                                                  pad=pad,
                                                  cover_all=cover_all,
                                                  dilate=dilate,
                                                  groups=groups,
                                                  cudnn_fast=cudnn_fast)
        self.binarization_fn = binarization_fn
        if pad != 0 and pad != (0, 0):
            self.pad = _pair(pad)
        else:
            self.pad = None

    def apply(self, inputs):
        # Unpack inputs
        x, W = inputs

        # Binarize weights
        W_b = self.binarization_fn(W)

        # Compute output
        y = super(BinaryConvolution2D, self).apply((x, W_b))
        return y


def binary_linear(x, W, binarization_fn=binarization_functions.deterministic_binarization, n_batch_axes=1):
    if n_batch_axes <= 0:
        raise ValueError('n_batch_axes should be greater than 0.')
    if n_batch_axes > 1:
        batch_shape = x.shape[:n_batch_axes]
        batch_size = utils.size_of_shape(batch_shape)
        x = x.reshape(batch_size, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    y, = BinaryLinear(binarization_fn=binarization_fn).apply((x, W))
    return y


def binary_convolution_2d(x, W,
                          stride=1,
                          pad=0,
                          cover_all=False,
                          binarization_fn=binarization_functions.deterministic_binarization,
                          **kwargs):
    y, = BinaryConvolution2D(stride=stride,
                             pad=pad,
                             cover_all=cover_all,
                             binarization_fn=binarization_fn,
                             **kwargs).apply((x, W))
    return y

