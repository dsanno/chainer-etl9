import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

CHARACTER_NUM = 3036

class Generator48(chainer.Chain):
    def __init__(self):
        super(Generator48, self).__init__(
            gy    = L.EmbedID(CHARACTER_NUM, 100),
            g1    = L.Linear(100, 6 * 6 * 256, wscale=0.02 * math.sqrt(100)),
            norm1 = L.BatchNormalization(6 * 6 * 256),
            g2    = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm2 = L.BatchNormalization(128),
            g3    = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm3 = L.BatchNormalization(64),
            g4    = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
        )

    def __call__(self, (z, y), train=True):
        h0 = z + self.gy(y)
        h1 = F.reshape(F.relu(self.norm1(self.g1(h0), test=not train)), (z.data.shape[0], 256, 6, 6))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.sigmoid(self.g4(h3))
        return h4

class Discriminator48(chainer.Chain):
    def __init__(self):
        super(Discriminator48, self).__init__(
            dc1   = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(64),
            dc2   = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            norm2 = L.BatchNormalization(128),
            dc3   = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm3 = L.BatchNormalization(256),
            dc4   = L.Linear(6 * 6 * 256, 100, wscale=0.02 * math.sqrt(6 * 6 * 256)),
            norm4 = L.BatchNormalization(100),
            dc5   = L.Linear(100, CHARACTER_NUM + 1, wscale=0.02 * math.sqrt(100)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        return self.dc5(h4)
