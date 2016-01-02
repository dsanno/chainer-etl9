# -*- coding: utf-8 -*-

import argparse
import numpy as np
from PIL import Image

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable, optimizers, serializers
from net import Generator48_2 as Generator

class Latent(chainer.Chain):
    def __init__(self, size):
        super(Latent, self).__init__(
            l1 = L.EmbedID(1, size)
        )

    def __call__(self, x):
        return self.l1(x)

parser = argparse.ArgumentParser(description='Chainer training example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', required=True, type=str,
                    help='model file path')
parser.add_argument('--input', '-i', required=True, type=str,
                    help='input image file path')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output image file path')
args = parser.parse_args()

gen = Generator()
serializers.load_hdf5(args.model, gen)

gpu_device = None
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu
    gen.to_gpu(gpu_device)
    xp = cuda.cupy
else:
    xp = np

index = 71 + 2010
image_size = 48
image_shape = (image_size, image_size)
latent_size = 100

t = 1 - xp.asarray(Image.open(args.input).convert('L').resize(image_shape)).reshape((1, 1) + image_shape).astype(np.float32) / 255
latent = Latent(latent_size)
if gpu_device >= 0:
    latent.to_gpu(gpu_device)
optimizer = optimizers.Adam()
optimizer.setup(latent)

s = Variable(xp.zeros((1,)).astype(np.int32))
y = Variable(xp.asarray([index]).astype(np.int32))
for i in range(2):
    z = latent(s)
    x = gen((z, y))
    loss = F.mean_squared_error(x, Variable(t))
    gen.zerograds()
    optimizer.zero_grads()
    loss.backward()
    print latent.l1.W.grad
    for n, p in gen.namedparams():
        print n
        print p.grad
    optimizer.update()
    print i, float(loss.data)

image_len = 100
s = Variable(xp.zeros((image_len,)).astype(np.int32), volatile=True)
#z = latent(s)
z = Variable(xp.random.uniform(-1, 1, (image_len, latent_size)).astype(np.float32), volatile=True)
#y = Variable(xp.asarray(xrange(0, 30 * image_len, 30)).astype(np.int32), volatile=True)
y = Variable(xp.asarray(xrange(index, index + image_len)).astype(np.int32), volatile=True)
x = gen((z, y), train=False)
image_array = ((1 - cuda.to_cpu(x.data)) * 255.99).astype(np.uint8).reshape((10, 10, 48, 48)).transpose((0, 2, 1, 3)).reshape((10 * 48, 10 * 48))
Image.fromarray(image_array).save(args.output)
