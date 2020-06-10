from chainer import Chain

from chainer import functions as F
from chainer import links as L


# noinspection PyAbstractClass
class Cifar10ConvNet(Chain):

    def __init__(self):
        super(Cifar10ConvNet, self).__init__()
        with self.init_scope():
            self.conv_1 = L.Convolution2D(in_channels=3, out_channels=32, ksize=3, pad=1)
            self.norm_1 = L.BatchNormalization(size=32)
            self.conv_2 = L.Convolution2D(in_channels=32, out_channels=32, ksize=3, pad=1)
            self.norm_2 = L.BatchNormalization(size=32)
            self.conv_3 = L.Convolution2D(in_channels=32, out_channels=64, ksize=3, pad=1)
            self.norm_3 = L.BatchNormalization(size=64)
            self.conv_4 = L.Convolution2D(in_channels=64, out_channels=64, ksize=3, pad=1)
            self.norm_4 = L.BatchNormalization(size=64)
            self.conv_5 = L.Convolution2D(in_channels=64, out_channels=128, ksize=3, pad=1)
            self.norm_5 = L.BatchNormalization(size=128)
            self.conv_6 = L.Convolution2D(in_channels=128, out_channels=128, ksize=3, pad=1)
            self.norm_6 = L.BatchNormalization(size=128)
            self.dense = L.Linear(in_size=2048, out_size=10)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.norm_1(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.norm_2(x)
        x = F.max_pooling_2d(x, 2)
        x = F.dropout(x, .2)
        x = self.conv_3(x)
        x = F.relu(x)
        x = self.norm_3(x)
        x = self.conv_4(x)
        x = F.relu(x)
        x = self.norm_4(x)
        x = F.max_pooling_2d(x, 2)
        x = F.dropout(x, .3)
        x = self.conv_5(x)
        x = F.relu(x)
        x = self.norm_5(x)
        x = self.conv_6(x)
        x = F.relu(x)
        x = self.norm_6(x)
        x = F.max_pooling_2d(x, 2)
        x = F.dropout(x, .4)
        x = self.dense(x)

        return x


