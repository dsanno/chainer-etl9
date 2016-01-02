import argparse
import numpy as np
import os
import cPickle as pickle

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from trainer import Trainer
from model import Classifier
from net import EtlNet

parser = argparse.ArgumentParser(description='Chainer training example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output model file path')
parser.add_argument('--iter', default=100, type=int,
                    help='number of iteration')

gpu_device = None
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu

model = Classifier(EtlNet())
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)

if args.input is not None:
    print 'input'
    serializers.load_hdf5(args.input + '.model', model)
    serializers.load_hdf5(args.input + '.state', optimizer)

state = {'max_accuracy': 0}
def progress_func(epoch, loss, accuracy, validate_loss, validate_accuracy, test_loss, test_accuracy):
    print('train    mean loss={}, accuracy={}'.format(loss, accuracy))
    if validate_loss is not None and validate_accuracy is not None:
        print('validate mean loss={}, accuracy={}'.format(validate_loss, validate_accuracy))
        if validate_accuracy > state['max_accuracy']:
            state['max_accuracy'] = validate_accuracy
    else:
        serializers.save_hdf5(args.output + '.model', model)
        serializers.save_hdf5(args.output + '.state', optimizer)
    if test_loss is not None and test_accuracy is not None:
        print('test     mean loss={}, accuracy={}'.format(test_loss, test_accuracy))

for i in range(args.iter):
    print 'epoch: {}'.format(i + 1)
    perm = np.random.permutation(45)
    for j in range(9):
        x, y = data.load(args.dataset_dir, perm[j * 5:(j + 1) * 5])
        Trainer.train(model, x, y, 1, batch_size=100, gpu_device=gpu_device, optimizer=optimizer, callback=progress_func)
    optimizer.lr *= 0.912
