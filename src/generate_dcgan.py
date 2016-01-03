# -*- coding: utf-8 -*-

import argparse
import numpy as np
from itertools import chain
from PIL import Image

from chainer import cuda, Variable, serializers
import net

def _flatten(l):
    return list(chain.from_iterable(l))

def jiscode_to_unicode(code):
    return ord('\x1b$B{0:c}{1:c}\x1b(B'.format(code / 256, code % 256).decode('iso-2022-jp'))

parser = argparse.ArgumentParser(description='Chainer training example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', required=True, type=str,
                    help='model file path')
parser.add_argument('--text', '-t', required=True, type=str,
                    help='output string')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output image file path')
parser.add_argument('--character_code', '-c', default='utf-8', type=str,
                    help='character code')
parser.add_argument('--size', '-s', default=96, type=int, choices=[48, 96],
                    help='image size')

args = parser.parse_args()
image_size = args.size

jis_codes = _flatten([
    [ 0x2422, 0x2424, 0x2426, 0x2428 ],
    range(0x242a, 0x2443),
    range(0x2444, 0x2463),
    [ 0x2464, 0x2466 ],
    range(0x2468, 0x246e),
    [ 0x246f, 0x2472, 0x2473],
    _flatten(map(lambda x: range(x + 0x21, x + 0x7f), range(0x3000, 0x4f00, 0x100))),
    range(0x4f21, 0x4f54),
])

codes = map(jiscode_to_unicode, jis_codes)
code_to_index = dict(zip(codes, range(len(codes))))

if image_size == 48:
    gen = net.Generator48()
else:
    gen = net.Generator()
serializers.load_hdf5(args.model, gen)

gpu_device = None
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu
    gen.to_gpu(gpu_device)
    xp = cuda.cupy
else:
    xp = np

latent_size = gen.latent_size
characters = map(lambda x: code_to_index[ord(x)], args.text.decode(args.character_code))
y = Variable(xp.asarray(characters).astype(np.int32), volatile=True)
z = Variable(xp.repeat(xp.random.uniform(-1, 1, (1, latent_size)), len(characters), axis=0).astype(np.float32), volatile=True)
x = gen((z, y), train=False)
n, ch, h, w = x.data.shape
image_array = ((1 - cuda.to_cpu(x.data)) * 256).clip(0, 255).astype(np.uint8).transpose((2, 0, 3, 1)).reshape((h, w * n))
Image.fromarray(image_array).save(args.output)
