import argparse
import numpy as np
import io
import os
import cPickle as pickle
from PIL import Image

import chainer
from chainer import cuda, optimizers, serializers, Variable
import chainer.links as L
from trainer import Trainer
from model import VAEModel
from net import EtlVAENet48Conv2 as EtlVAENet
import data

parser = argparse.ArgumentParser(description='Chainer training example: ETL9')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output model file path')
parser.add_argument('--iter', default=100, type=int,
                    help='number of iteration')
parser.add_argument('--dataset', '-d', default='dataset/etl9g.pkl', type=str,
                    help='dataset file path')
parser.add_argument('--out_image_dir', default=None, type=str,
                    help='directory to output test images')

args = parser.parse_args()
if args.out_image_dir != None:
    if not os.path.exists(args.out_image_dir):
        try:
            os.mkdir(args.out_image_dir)
        except:
            print 'cannot make directory {}'.format(args.out_image_dir)
            exit()
    elif not os.path.isdir(args.out_image_dir):
        print 'file path {} exists but is not directory'.format(args.out_image_dir)
        exit()

if args.gpu >= 0:
    cuda.check_cuda_available()

with open(args.dataset, 'rb') as f:
    images, labels = pickle.load(f)

model = VAEModel(EtlVAENet())
optimizer = optimizers.Adam(alpha=0.0002, beta1=0.5)
#optimizer = optimizers.Adam()
optimizer.setup(model)

if args.input is not None:
    serializers.load_hdf5(args.input + '.model', model)
    serializers.load_hdf5(args.input + '.state', optimizer)

state = {'max_accuracy': 0}
def progress_func(epoch, loss, accuracy, validate_loss, validate_accuracy, test_loss, test_accuracy):
    print('train    mean loss={}, accuracy={}'.format(loss, accuracy))
    if validate_loss is not None and validate_accuracy is not None:
        print('validate mean loss={}, accuracy={}'.format(validate_loss, validate_accuracy))
    if test_loss is not None and test_accuracy is not None:
        print('test     mean loss={}, accuracy={}'.format(test_loss, test_accuracy))

def train_one(train_model, train_optimizer, x_batch, y_batch, gpu_device):
    if gpu_device != None:
        x = Variable(cuda.cupy.asarray(x_batch))
        y = Variable(cuda.cupy.asarray(y_batch))
    else:
        x = Variable(x_batch)
        y = Variable(y_batch)
    train_optimizer.update(train_model, (x, y), x)
    return (float(model.loss.data), float(model.accuracy.data))

image_save_interval = 20000
def train(train_model, train_optimizer, x_train, y_train, epoch_num, batch_size, gpu_device):
    if gpu_device == None:
        xp = np
        train_model.to_cpu()
    else:
        xp = cuda.cupy
        train_model.to_gpu(gpu_device)
    x_size = len(x_train)
    x_batch = np.zeros((batch_size, 1, 48, 48), dtype=np.float32)
    for epoch in range(epoch_num):
        perm = np.random.permutation(x_size)
        loss_sum = 0
        accuracy_sum = 0
        for i in xrange(0, x_size, batch_size):
            x_batch.fill(0)
            for j in range(batch_size):
                with io.BytesIO(x_train[i + j]) as b:
                    offset_x = np.random.randint(4)
                    offset_y = np.random.randint(4)
                    pixels = np.asarray(Image.open(b).convert('L').resize((48, 48))).astype(np.float32).reshape((1, 48, 48))
                    x_batch[j, :, offset_y:offset_y + 45, offset_x:offset_x + 45] = pixels[:,3:48,2:47] / 255
            ##
            #x_batch = x_batch.reshape((batch_size, 48 * 48))
            ##
            y_batch = y_train[perm[i:i + batch_size]]
            loss, accuracy = train_one(train_model, train_optimizer, x_batch, y_batch, gpu_device)
            loss_sum += loss * batch_size
            accuracy_sum += accuracy * batch_size

            if i % image_save_interval == 0 and args.out_image_dir != None:
                print '{} {}'.format(loss_sum / (i + batch_size), accuracy_sum / (i + batch_size))
                test_size = 10
                x_rec = xp.zeros((test_size, 1, 48, 48)).astype(np.float32)
#                x_rec = xp.zeros((test_size, 48 * 48)).astype(np.float32)
                for j in range(test_size):
                    with io.BytesIO(x_train[j]) as b:
                        pixels = xp.asarray(Image.open(b).convert('L').resize((48, 48))).astype(np.float32).reshape((1, 48, 48)) / 255
#                        pixels = xp.asarray(Image.open(b).convert('L').resize((48, 48))).astype(np.float32).reshape((48 * 48,)) / 255
                        x_rec[j] = pixels
                y_rec = xp.asarray(y_train[:test_size])
                y_gen = xp.asarray(xrange(0, test_size * 100, 100)).astype(np.int32)
                test_images = train_model.predictor.generate(Variable(x_rec), Variable(y_rec), Variable(y_gen))
                test_images = ((1 - cuda.to_cpu(test_images.data)) * 255).clip(0, 255).astype(np.uint8)
                test_images = test_images.reshape((test_size, test_size, 48, 48)).transpose((0, 2, 1, 3)).reshape((test_size * 48, test_size * 48))
                Image.fromarray(test_images).save('{0}/{1:03d}_{2:07d}.png'.format(args.out_image_dir, epoch, i))

        serializers.save_hdf5('{0}_{1:03d}.model'.format(args.output, epoch), model)
        serializers.save_hdf5('{0}_{1:03d}.state'.format(args.output, epoch), optimizer)
        print 'epoch {} done'.format(epoch)
        print 'loss:     {}'.format(loss_sum / x_size)
        print 'accuracy: {}'.format(accuracy_sum / x_size)

train(model, optimizer, images, labels.astype(np.int32), 100, 100, args.gpu)
