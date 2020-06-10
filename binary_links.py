import typing as tp
import numpy as np

import chainer
from chainer import types, initializers, variable, utils, memory_layouts
from chainer import link as L
from chainer.utils import argument

import binarization_functions
import binary_functions as BF


class BinaryLinear(L.Link):

    def __init__(self,
                 in_size: tp.Optional[int]
                    = None,
                 out_size: tp.Optional[int]
                    = None,
                 initialW: tp.Optional[types.InitializerSpec]
                    = None,
                 binarization_fn: tp.Callable[[variable.Variable], variable.Variable]
                    = binarization_functions.deterministic_binarization,
                 gain: bool = False,
                 ):

        super(BinaryLinear, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.in_size = in_size
        self.out_size = out_size
        self.binarization_fn = binarization_fn
        self.gain = gain

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)  # type: variable.Parameter
            self.W.binary = True
            if self.gain:
                self.y = variable.Parameter(initializers.One(), shape=(1,))
            if in_size is not None:
                self._initialize_params(in_size)

    def _initialize_params(self, in_size: int) -> None:
        self.W.initialize((self.out_size, in_size))

    @property
    def printable_specs(self):
        specs = [
            ('in_size', self.in_size),
            ('out_size', self.out_size),
        ]
        for spec in specs:
            yield spec

    @classmethod
    def from_params(cls, W):
        """Initialize a :class:`~chainer.links.Linear` with given parameters.

        This method uses ``W`` and optional ``b`` to initialize a linear layer.

        Args:
            W (:class:`~chainer.Variable` or :ref:`ndarray`):
                The weight parameter.
        """
        out_size, in_size = W.shape
        link = cls(in_size, out_size, initialW=variable.as_array(W))
        return link

    def forward(
            self,
            x: variable.Variable,
            n_batch_axes: int = 1
    ) -> variable.Variable:
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.
            n_batch_axes (int): The number of batch axes. The default is 1. The
                input variable is reshaped into
                (:math:`{\\rm n\\_batch\\_axes} + 1`)-dimensional tensor.
                This should be greater than 0.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        if self.W.array is None:
            in_size = utils.size_of_shape(x.shape[n_batch_axes:])
            self._initialize_params(in_size)
        o = BF.binary_linear(x, self.W, n_batch_axes=n_batch_axes)

        if self.gain:
            o = self.y * o
        return o


class BinaryConvolution2D(L.Link):

    def __init__(self,
                 in_channels: tp.Optional[int],
                 out_channels: int,
                 ksize: tp.Union[int, tp.Tuple[int, int]],
                 stride: int = 1,
                 pad: int = 0,
                 initialW: tp.Optional[types.InitializerSpec]
                    = None,
                 binarization_fn: tp.Callable[[variable.Variable], variable.Variable]
                    = binarization_functions.deterministic_binarization,
                 gain: bool = False,
                 **kwargs):
        super(BinaryConvolution2D, self).__init__()

        dilate, groups = argument.parse_kwargs(
            kwargs, ('dilate', 1), ('groups', 1),
            deterministic='deterministic argument is not supported anymore. '
                          'Use chainer.using_config(\'cudnn_deterministic\', value) '
                          'context where value is either `True` or `False`.')

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.cudnn_fast = chainer.get_compute_mode() == 'cudnn_fast'
        if self.cudnn_fast:
            x_layout = memory_layouts.CUDNN_CHANNEL_LAST_X
            w_layout = memory_layouts.CUDNN_CHANNEL_LAST_W
        else:
            x_layout = memory_layouts.CUDNN_CHANNEL_FIRST_X
            w_layout = memory_layouts.CUDNN_CHANNEL_FIRST_W

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.dilate = _pair(dilate)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = int(groups)
        self.x_layout = x_layout
        self.binarization_fn = binarization_fn
        self.binary = True
        self.gain = gain

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer, layout=w_layout)
            self.W.binary = True
            if self.gain:
                self.y = variable.Parameter(initializers.One(), shape=(1,))
            if in_channels is not None:
                self._initialize_params(in_channels)

    @property
    def printable_specs(self):
        specs = [
            ('in_channels', self.in_channels),
            ('out_channels', self.out_channels),
            ('ksize', self.ksize),
            ('stride', self.stride),
            ('pad', self.pad),
            ('dilate', self.dilate),
            ('groups', self.groups),
        ]
        for spec in specs:
            yield spec

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        if self.out_channels % self.groups != 0:
            raise ValueError('the number of output channels must be'
                             ' divisible by the number of groups')
        if in_channels % self.groups != 0:
            raise ValueError('the number of input channels must be'
                             ' divisible by the number of groups')
        W_shape = (self.out_channels, int(in_channels / self.groups), kh, kw)
        self.W.initialize(W_shape)

    @classmethod
    def from_params(cls, W, stride=1, pad=0, **kwargs):
        dilate, groups = argument.parse_kwargs(
            kwargs, ('dilate', 1), ('groups', 1))
        out_channels, _in_channels, kw, kh = W.shape
        in_channels = _in_channels * groups

        link = cls(
            in_channels, out_channels, (kw, kh), stride, pad,
            initialW=variable.as_array(W),
            dilate=dilate,
            groups=groups)
        return link

    def forward(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.
        """
        x = chainer.as_variable(x)
        assert x.layout == self.x_layout

        if not self.W.is_initialized:
            _, c, _, _ = memory_layouts.get_semantic_shape(
                x, assumed_layout=self.x_layout)
            self._initialize_params(c)

        out = BF.binary_convolution_2d(
            x, self.W, stride=self.stride, pad=self.pad, dilate=self.dilate,
            groups=self.groups, cudnn_fast=self.cudnn_fast)
        if self.gain:
            out = self.y * out

        return out


class ResidualBinarization(L.Link):
    def __init__(self, n_levels=None):
        super(ResidualBinarization, self).__init__()
        with self.init_scope():
            self.levels = chainer.Parameter(np.power(.5, np.arange(n_levels, dtype=np.float32)))

    def __call__(self, x):
        return binarization_functions.residual_binarization(x, self.levels)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x




