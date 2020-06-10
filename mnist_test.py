from chainer import ChainList
from chainer import functions as F
from chainer import links as L, iterators, optimizers, training
from chainer.datasets import mnist
from chainer.training import extensions

import binary_chains
from hooks import BinaryWeightClip


def test_dense_real():
    class Net(ChainList):
        def __init__(self):
            super(Net, self).__init__()
            self.l1 = L.Linear(784, 4096)
            self.bn1 = L.BatchNormalization(4096)
            self.l2 = L.Linear(4096, 4096)
            self.bn2 = L.BatchNormalization(4096)
            self.l3 = L.Linear(4096, 4096)
            self.bn3 = L.BatchNormalization(4096)
            self.l4 = L.Linear(4096, 10)
            self.add_link(self.l1)
            self.add_link(self.l2)
            self.add_link(self.l3)
            self.add_link(self.l4)
            self.add_link(self.bn1)
            self.add_link(self.bn2)
            self.add_link(self.bn3)

        def forward(self, x):
            x = self.l1(x)
            x = self.bn1(x)
            x = F.sigmoid(x)
            x = self.l2(x)
            x = self.bn2(x)
            x = F.sigmoid(x)
            x = self.l3(x)
            x = self.bn3(x)
            x = F.sigmoid(x)
            x = self.l4(x)
            return x

    train, test = mnist.get_mnist(ndim=1)

    batchsize = 128

    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, False, False)

    gpu_id = 0

    model = L.Classifier(Net())
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    max_epoch = 100

    optimizer = optimizers.Adam(alpha=0.0001)
    optimizer.setup(model)
    optimizer.add_hook(BinaryWeightClip())

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')
    trainer.extend(extensions.LogReport())
    # trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    # trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss_gain.png'))
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy_gain.png'))
    trainer.extend(extensions.DumpGraph('main/loss'))

    trainer.run()


def test_bnn():


    train, test = mnist.get_mnist(ndim=1)

    batchsize = 128

    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, False, False)

    gpu_id = 0

    model = L.Classifier(binary_chains.CouvBNN())
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    max_epoch = 100

    optimizer = optimizers.Adam(alpha=0.0001)
    optimizer.setup(model)
    optimizer.add_hook(BinaryWeightClip())

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')
    trainer.extend(extensions.LogReport())
    # trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    # trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss_gain.png'))
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy_gain.png'))
    trainer.extend(extensions.DumpGraph('main/loss'))

    trainer.run()


def test_binary_cnn():
    class BinLeNet(ChainList):
        def __init__(self):
            gain = True
            super(BinLeNet, self).__init__(binary_chains.BNNConvChain(in_channels=1, out_channels=6, ksize=5, stride=1,
                                                                      pooling_size=2, activation_fn=None,
                                                                      gain=gain, prelu=gain, residual_levels=4),
                                           binary_chains.BNNConvChain(in_channels=6, out_channels=16, ksize=5, stride=1,
                                                                      pooling_size=2,
                                                                      gain=gain, prelu=gain, residual_levels=4),
                                           binary_chains.BNNConvChain(in_channels=16, out_channels=120, ksize=4, stride=1,
                                                                      pooling=None,
                                                                      gain=gain, prelu=gain, residual_levels=4),
                                           binary_chains.BNNDenseChain(120, 84,
                                                                       gain=gain, prelu=gain, residual_levels=4),
                                           binary_chains.BNNDenseChain(84, 10,
                                                                       gain=gain, prelu=gain, residual_levels=4),
                                           L.Scale(W_shape=(10,))
                                           )

        def forward(self, x):
            for f in self.children():
                x = f(x)
            return x

    train, test = mnist.get_mnist(ndim=3)

    batchsize = 128

    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, False, False)

    gpu_id = 0

    model = L.Classifier(BinLeNet())
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    max_epoch = 100

    optimizer = optimizers.Adam(alpha=0.0001)
    optimizer.setup(model)
    optimizer.add_hook(BinaryWeightClip())

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')
    trainer.extend(extensions.LogReport(filename='log_gain'))
    # trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    # trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss_gain.png'))
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy_gain.png'))
    trainer.extend(extensions.DumpGraph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    test_binary_cnn()
