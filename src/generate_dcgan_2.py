# -*- coding: utf-8 -*-

import argparse
import numpy as np
from PIL import Image

import chainer
from chainer import cuda, Variable, serializers
from net import Generator2

parser = argparse.ArgumentParser(description='Chainer training example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input', '-i', required=True, type=str,
                    help='input model file path')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output image file path')
args = parser.parse_args()

gen = Generator2()
serializers.load_hdf5(args.input, gen)

gpu_device = None
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu
    gen.to_gpu(gpu_device)
    xp = cuda.cupy
else:
    xp = np

LATENT_SIZE = 100
image_len = 100
#z = Variable(xp.zeros((image_len, LATENT_SIZE)).astype(np.float32))
z = Variable(xp.random.uniform(-1, 1, (image_len, LATENT_SIZE)).astype(np.float32))
y = Variable(xp.asarray(xrange(0, 30 * image_len, 30)).astype(np.int32))
x = gen((z, y), train=False)
image_array = ((1 - cuda.to_cpu(x.data)) * 255.99).astype(np.uint8).reshape((10, 10, 96, 96)).transpose((0, 2, 1, 3)).reshape((10 * 96, 10 * 96))
Image.fromarray(image_array).save(args.output)
