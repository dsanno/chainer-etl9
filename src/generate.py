import os
import numpy as np
import argparse
import cPickle as pickle
from scipy import misc
from chainer import cuda, Variable, serializers
from chainer_trainer.model import VAEModel
from net import EtlVAENet

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i',      type=str, required=True,
                    help="input model file path")
parser.add_argument('--output_dir', '-o', type=str, default="generated",
                    help="output directory path")
parser.add_argument('--gpu', '-g',        type=int, default=-1,
                    help="GPU ID (negative value indicates CPU)")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
model = VAEModel(EtlVAENet())
serializers.load_hdf5(args.input, model)
predictor = model['predictor']

if args.gpu >= 0:
    model.to_gpu(args.gpu)
    xp = cuda.cupy
else:
    model.to_cpu()
    xp = np

def load_data(index):
    path = os.path.join('dataset', 'etl9_{0:02d}.pkl'.format(index))
    print 'Loading {}...'.format(path)
    with open(path, 'rb') as f:
        data, labels = pickle.load(f)
        return (data.astype(np.float32) / 15, labels.astype(np.int32))

x_test, y_test = load_data(0)

category_num = 3036
perm = np.random.permutation(len(y_test))
sample_num = 20
#sample_category = 10
sample_category = 1
W = 128
H = 127

for i in range(sample_num):
    j = perm[i]
    x = Variable(xp.asarray(x_test[j:j + 1]))
    y_rec = Variable(xp.asarray(y_test[j:j + 1]))
#    y_gen = Variable(xp.asarray(np.random.permutation(category_num)[:sample_category]).astype(np.int32))
    y_gen = Variable(xp.asarray([1]).astype(np.int32))
    y = predictor.generate(x, y_rec, y_gen)
    y_data = cuda.to_cpu(y.data)
    x_data = cuda.to_cpu(x.data)
    image = 1.0 - np.vstack((x_data, y_data)).reshape(sample_category + 1, H, W).swapaxes(0, 1).reshape(H, (sample_category + 1) * W)
    misc.imsave('{}/{}.jpg'.format(args.output_dir, i), image)
