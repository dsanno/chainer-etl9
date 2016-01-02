import argparse
import numpy as np
import io
import os
from PIL import Image
import cPickle as pickle

import chainer
from chainer import cuda, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L
from net import Generator48 as Generator, Discriminator48 as Discriminator

parser = argparse.ArgumentParser(description='Chainer training example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output model file path without extension')
parser.add_argument('--iter', default=100, type=int,
                    help='number of iteration')
parser.add_argument('--out_image_dir', default=None, type=str,
                    help='output directory to output images')
parser.add_argument('--dataset', '-d', default='dataset/etl9g.pkl', type=str,
                    help='dataset file path')
args = parser.parse_args()

gen_model = Generator()
optimizer_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_gen.setup(gen_model)
optimizer_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
dis_model = Discriminator()
optimizer_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_dis.setup(dis_model)
optimizer_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

if args.input != None:
    serializers.load_hdf5(args.input + '.gen.model', gen_model)
    serializers.load_hdf5(args.input + '.gen.state', optimizer_gen)
    serializers.load_hdf5(args.input + '.dis.model', dis_model)
    serializers.load_hdf5(args.input + '.dis.state', optimizer_dis)

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

with open(args.dataset, 'rb') as f:
    images, labels = pickle.load(f)
    labels = labels.astype(np.int32)

gpu_device = None
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu

LATENT_SIZE = 100
BATCH_SIZE = 100
image_save_interval = 50000

def train_gen(gen, dis, optimizer_gen, optimizer_dis, x_batch, gpu_device):
    batch_size = len(x_batch)
    if gpu_device == None:
        xp = xp
    else:
        xp = cuda.cupy
    z = Variable(xp.random.uniform(-1, 1, (batch_size, LATENT_SIZE)).astype(np.float32))
    x = gen(z)
    y1 = dis(x)
    loss_gen = F.softmax_cross_entropy(y1, Variable(xp.zeros(batch_size).astype(np.int32)))
    loss_dis = F.softmax_cross_entropy(y1, Variable(xp.ones(batch_size).astype(np.int32)))
    optimizer_gen.zero_grads()
    loss_gen.backward()
    optimizer_gen.update()
    return loss_gen.data

def train_dis(gen, dis, optimizer_gen, optimizer_dis, x_batch, gpu_device):
    batch_size = len(x_batch)
    if gpu_device == None:
        xp = xp
    else:
        xp = cuda.cupy
    z = Variable(xp.random.uniform(-1, 1, (batch_size, LATENT_SIZE)).astype(np.float32))
    x = gen(z)
    y1 = dis(x)
    loss_dis = F.softmax_cross_entropy(y1, Variable(xp.ones(batch_size).astype(np.int32)))
    y2 = dis(Variable(xp.asarray(x_batch)))
    loss_dis += F.softmax_cross_entropy(y2, Variable(xp.zeros(batch_size).astype(np.int32)))
    optimizer_dis.zero_grads()
    loss_dis.backward()
    optimizer_dis.update()
    return loss_dis.data

def train_one(gen, dis, optimizer_gen, optimizer_dis, x_batch, y_batch, gpu_device):
    batch_size = len(x_batch)
    if gpu_device == None:
        xp = xp
    else:
        xp = cuda.cupy
    # train generator
    y = Variable(xp.asarray(y_batch))
    t = Variable(xp.asarray(y_batch + 1))
    z = Variable(xp.random.uniform(-1, 1, (batch_size, LATENT_SIZE)).astype(np.float32))
    x = gen((z, y))
    y1 = dis(x)
    loss_gen = F.softmax_cross_entropy(y1, t)
    loss_dis = F.softmax_cross_entropy(y1, Variable(xp.zeros(batch_size).astype(np.int32)))
    # train discriminator
    y2 = dis(Variable(xp.asarray(x_batch)))
    loss_dis += F.softmax_cross_entropy(y2, t)

    optimizer_gen.zero_grads()
    loss_gen.backward()
    optimizer_gen.update()

    optimizer_dis.zero_grads()
    loss_dis.backward()
    optimizer_dis.update()

    return (float(loss_gen.data), float(loss_dis.data))

def train(gen, dis, optimizer_gen, optimizer_dis, epoch_num, gpu_device=None, out_image_dir=None):
    if gpu_device == None:
        gen.to_cpu()
        dis.to_cpu()
        xp = np
    else:
        gen.to_gpu(gpu_device)
        dis.to_gpu(gpu_device)
        xp = cuda.cupy
    out_image_len = 100
    z_out_image =  Variable(xp.random.uniform(-1, 1, (out_image_len, LATENT_SIZE)).astype(np.float32))
    x_batch = np.zeros((BATCH_SIZE, 1, 48, 48), dtype=np.float32)
    for epoch in xrange(1, epoch_num + 1):
        x_size = len(images)
        perm = np.random.permutation(x_size)
        sum_loss_gen = 0
        sum_loss_dis = 0
        for i in xrange(0, x_size, BATCH_SIZE):
            x_batch.fill(0)
            for j, p in enumerate(perm[i:i + BATCH_SIZE]):
                image = images[p]
                offset_x = np.random.randint(4)
                offset_y = np.random.randint(4)
                with io.BytesIO(image) as b:
                    pixels = np.asarray(Image.open(b).convert('L').resize((48, 48))).astype(np.float32).reshape((1, 48, 48))
                    x_batch[j, :, offset_y:offset_y + 45, offset_x:offset_x + 45] = pixels[:,3:48,2:47] / 255
            y_batch = labels[perm[i:i + BATCH_SIZE]]
            loss_gen, loss_dis = train_one(gen, dis, optimizer_gen, optimizer_dis, x_batch, y_batch, gpu_device)
            sum_loss_gen += loss_gen * BATCH_SIZE
            sum_loss_dis += loss_dis * BATCH_SIZE
            if i % image_save_interval == 0:
                print '{} {}'.format(sum_loss_gen / (i + BATCH_SIZE), sum_loss_dis / (i + BATCH_SIZE))
                if out_image_dir != None:
                    y_gen = Variable(xp.asarray(xrange(0, 3000, 30)).astype(np.int32))
                    data = gen((z_out_image, y_gen), train=False).data
                    image = ((1 - cuda.to_cpu(data)) * 255.99).astype(np.uint8).reshape((10, 10, 48, 48)).transpose((0, 2, 1, 3)).reshape((10 * 48, 10 * 48))
                    Image.fromarray(image).save('{0}/{1:03d}_{2:07d}.png'.format(out_image_dir, epoch, i))
        print 'epoch: {} done'.format(epoch)
        print('gen loss={}'.format(sum_loss_gen / x_size))
        print('dis loss={}'.format(sum_loss_dis / x_size))
        serializers.save_hdf5('{0}_{1:03d}.gen.model'.format(args.output, epoch), gen)
        serializers.save_hdf5('{0}_{1:03d}.gen.state'.format(args.output, epoch), optimizer_gen)
        serializers.save_hdf5('{0}_{1:03d}.dis.model'.format(args.output, epoch), dis)
        serializers.save_hdf5('{0}_{1:03d}.dis.state'.format(args.output, epoch), optimizer_dis)

train(gen_model, dis_model, optimizer_gen, optimizer_dis, args.iter, gpu_device=gpu_device, out_image_dir=args.out_image_dir)
