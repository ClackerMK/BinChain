import chainer
from chainer import links as L, optimizers, iterators, training
from chainer.datasets import get_cifar10, get_mnist
from chainer.links import VGG16Layers
from chainer.training import extensions

import binary_chains
import real_networks
from hooks import BinaryWeightClip

from experimentator import Experiment, order


def train_network(network, name, train, test, reload=True, max_epoch=100, binary=True, alpha=0.0001):
    batchsize = 128

    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, False, False)

    gpu_id = 0

    model = L.Classifier(network)
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    if alpha is not None:
        optimizer = optimizers.Adam(alpha=0.0001)
    else:
        optimizer = optimizers.Adam()

    optimizer.setup(model)
    if binary:
        optimizer.add_hook(BinaryWeightClip())

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out=name)

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}',
                                       n_retains=1,
                                       autoload=reload))
    trainer.extend(extensions.snapshot_object(model.predictor,
                                              filename='model_epoch-{.updater.epoch}',
                                              n_retains=1,
                                              autoload=reload))
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch',
                              file_name='accuracy.png'))
    trainer.extend(extensions.DumpGraph('main/loss'))

    trainer.run()

#
# def compare_residual_binarization_bnn_mnist():
#     train, test = get_mnist()
#     parent_dir = "results/MNIST/BNN/"
#
#     train_network(BNN.CouvBNN(nogain=True, noprelu=True), parent_dir + "nogain", train, test)
#     train_network(BNN.CouvBNN(), parent_dir + "gain", train, test)
#
#     for l in range(2, 5):
#         bnn_residual = BNN.CouvBNN(residual_levels=l)
#         train_network(bnn_residual, parent_dir + "residual{}".format(l), train, test)
#
#
# def compare_residual_binarization_mnist():
#     train, test = get_mnist()
#     parent_dir = "results/CIFAR-10/Conv/"
#
#     train_network(BNN.BinLeNet(nogain=True, noprelu=True), parent_dir + "nogain", train, test)
#     train_network(BNN.BinCifar10ConvNet(residual_levels=0),         parent_dir + "gain", train, test)
#     for i in range(3):
#         train_network(BNN.BinCifar10ConvNet(residual_levels=i+2),   parent_dir + "residual{}".format(i), train, test)
#
#     train_network(real_networks.Cifar10ConvNet(), parent_dir + "real", train, test, binary=False)
#
#
# def compare_residual_binarization_cifar_01():
#     train, test = get_cifar10()
#     parent_dir = "results/CIFAR-10/Conv_001/"
#
#     train_network(BNN.BinCifar10ConvNet(nogain=True, noprelu=True), parent_dir + "nogain", train, test, alpha=0.01)
#     train_network(BNN.BinCifar10ConvNet(residual_levels=0),         parent_dir + "gain", train, test, alpha=0.01)
#     train_network(BNN.BinCifar10ConvNet(residual_levels=2),         parent_dir + "residual2", train, test, alpha=0.01)
#
#     train_network(real_networks.Cifar10ConvNet(), parent_dir + "real", train, test, binary=False, alpha=0.01)
#
#
# def compare_binarization_prelu_less_mnist():
#     train, test = get_cifar10()
#     parent_dir = "results/MNIST/Con_/"
#
#     train_network(BNN.BinLeNet(nogain=False, noprelu=True),    parent_dir + "noprelu_gain", train, test)
#
#
# def optimize_binary_lenet_mnist():
#     train, test = get_mnist(ndim=3)
#     parent_dir = "results/MNIST/LeNet/"
#
#     net = BNN.BinLeNet(residual_levels=4)
#     train_network(net, parent_dir + "residual{}".format(4), train, test)


if __name__ == "__main__":
    # Load MNIST 1d
    train, test = get_mnist()

    # Test Cases
    learn_rate_terms = {
        'slow': 0.0001,
        'fast': 0.01
    }

    for name, case in learn_rate_terms.items():
        train_network(binary_chains.CouvBNN(residual_levels=4), 'results/mnist/dense/' + name, train, test, alpha=case)

    # Load MNIST 3d
    train, test = get_mnist(ndim=3)
    # Test Cases
    gain_terms = {
        'nogain_noprelu':   {'nogain': True, 'noprelu': True},
        'gain_noprelu':     {'nogain': False, 'noprelu': True},
        'gain_prelu':       {'nogain': True, 'noprelu': True},
        'residual2':        {'residual_levels': 2},
        'residual3':        {'residual_levels': 3},
        'residual4':        {'residual_levels': 4}
    }

    for name, case in gain_terms.items():
        train_network(binary_chains.BinLeNet(**case), 'results/mnist/binLe/' + name, train, test)

    # Load Cifar-10
    train, test = get_cifar10(ndim=3)

    # Test Case real vs residuals
    train_network(binary_chains.BinCifar10ConvNet(residual_levels=2),    'results/cifar-10/residual2', train, test)
    train_network(binary_chains.BinCifar10ConvNet(residual_levels=3),    'results/cifar-10/residual3', train, test)
    train_network(binary_chains.BinCifar10ConvNet(residual_levels=4),    'results/cifar-10/residual4', train, test)
    train_network(real_networks.Cifar10ConvNet(),                        'results/cifar-10/real', train, test)









