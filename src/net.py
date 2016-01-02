import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

WIDTH = 96
HEIGHT = 96
IN_SIZE = WIDTH * HEIGHT
CHARACTER_NUM = 3036

class ConvPool(chainer.Chain):
    def __init__(self, channel_num, activator=F.leaky_relu):
        super(ConvPool, self).__init__(
            conv1 = L.Convolution2D(channel_num, channel_num, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * channel_num)),
            norm1 = L.BatchNormalization(channel_num),
            conv2 = L.Convolution2D(channel_num, channel_num, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * channel_num)),
            norm2 = L.BatchNormalization(channel_num),
        )
        self.activator=activator

    def __call__(self, x, train=True):
        h1 = self.activator(self.norm1(self.conv1(x), test=not train)) + F.average_pooling_2d(x, 2)
        h2 = self.activator(self.norm2(self.conv2(x), test=not train)) + F.average_pooling_2d(x, 2)
        return F.concat([h1, h2])

class DeconvPool(chainer.Chain):
    def __init__(self, channel_num, activator=F.relu):
        assert channel_num % 2 == 0
        super(DeconvPool, self).__init__(
            deconv1 = L.Deconvolution2D(channel_num / 2, channel_num / 2, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * channel_num / 2)),
            norm1 = L.BatchNormalization(channel_num),
            deconv2 = L.Deconvolution2D(channel_num / 2, channel_num / 2, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * channel_num / 2)),
            norm2 = L.BatchNormalization(channel_num),
        )
        self.out_channel_num = channel_num / 2
        self.activator = activator

    def __call__(self, x, train=True):
        x1, x2 = F.split_axis(x, self.out_channel_num, 1)
        h1 = self.activator(self.norm1(self.deconv1(x1), test=not train)) + x1
        h2 = self.activator(self.norm2(self.deconv2(x2), test=not train)) + x2
        return h1 + h2

class EtlNet(chainer.Chain):
    def __init__(self):
        super(EtlNet, self).__init__(
            l1 = L.Linear(IN_SIZE, 4096),
            l2 = L.Linear(4096, 4096),
            l3 = L.Linear(4096, CHARACTER_NUM)
        )

    def __call__(self, x_var, train=True):
        h1 = F.dropout(F.relu(self.l1(x_var)), train=train)
        h2 = F.dropout(F.relu(self.l2(h1)), train=train)
        return self.l3(h2)

class EtlConvNet(chainer.Chain):
    def __init__(self):
        super(EtlConvNet, self).__init__(
            l1 = L.Convolution2D(1, 20, 3, pad=1),
            l2 = L.Convolution2D(20, 40, 3, pad=1),
            l3 = L.Convolution2D(40, 80, 3, pad=1),
            l4 = L.Convolution2D(80, 160, 3, pad=1),
            l5 = L.Linear(10240, 4096),
            l6 = L.Linear(4096, CHARACTER_NUM)
        )

    def __call__(self, x, train=True):
        h1 = F.max_pooling_2d(F.relu(self.l1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.l2(h1)), 2)
        h3 = F.max_pooling_2d(F.relu(self.l3(h2)), 2)
        h4 = F.max_pooling_2d(F.relu(self.l4(h3)), 2)
        h5 = F.relu(self.l5(h4))
        h6 = F.relu(self.l6(h5))
        return h6

class EtlVAENet(chainer.Chain):
    def __init__(self):
        super(EtlVAENet, self).__init__(
            r1     = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            rnorm1 = L.BatchNormalization(64),
            r2     = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            rnorm2 = L.BatchNormalization(128),
            r3     = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            rnorm3 = L.BatchNormalization(256),
            r4     = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            rnorm4 = L.BatchNormalization(512),
            mean   = L.Linear(6 * 6 * 512, 100, wscale=0.02 * math.sqrt(6 * 6 * 512)),
            var    = L.Linear(6 * 6 * 512, 100, wscale=0.02 * math.sqrt(6 * 6 * 512)),
            gy     = L.EmbedID(CHARACTER_NUM, 100),
            g1     = L.Linear(100, 6 * 6 * 512, wscale=0.02 * math.sqrt(100)),
            gnorm1 = L.BatchNormalization(6 * 6 * 512),
            g2     = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 512)),
            gnorm2 = L.BatchNormalization(256),
            g3     = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            gnorm3 = L.BatchNormalization(128),
            g4     = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            gnorm4 = L.BatchNormalization(64),
            g5     = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
        )

    def __call__(self, (x, y), train=True):
        xp = cuda.get_array_module(x.data)
        r1 = F.leaky_relu(self.rnorm1(self.r1(x), test=not train))
        r2 = F.leaky_relu(self.rnorm2(self.r2(r1), test=not train))
        r3 = F.leaky_relu(self.rnorm3(self.r3(r2), test=not train))
        r4 = F.leaky_relu(self.rnorm4(self.r4(r3), test=not train))
        mean = self.mean(r4) - self.gy(y)
        var = 0.5 * self.var(r4)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + self.gy(y) + F.exp(var) * Variable(rand, volatile=not train)
        g1 = F.reshape(F.relu(self.gnorm1(self.g1(z), test=not train)), (z.data.shape[0], 512, 6, 6))
        g2 = F.relu(self.gnorm2(self.g2(g1), test=not train))
        g3 = F.relu(self.gnorm3(self.g3(g2), test=not train))
        g4 = F.relu(self.gnorm4(self.g4(g3), test=not train))
        g5 = F.sigmoid(self.g5(g4))
        return (g5, mean, var)

    def generate(self, x_rec, y_rec, y_gen):
        assert x_rec.data.shape[0] == y_rec.data.shape[0]
        rec_num = x_rec.data.shape[0]
        gen_num = y_gen.data.shape[0]
        xp = cuda.get_array_module(x_rec.data)
        r1 = F.leaky_relu(self.rnorm1(self.r1(x_rec), test=True))
        r2 = F.leaky_relu(self.rnorm2(self.r2(r1), test=True))
        r3 = F.leaky_relu(self.rnorm3(self.r3(r2), test=True))
        r4 = F.leaky_relu(self.rnorm4(self.r4(r3), test=True))
        mean = self.mean(r4) - self.gy(y_rec)
        var = 0.5 * self.var(r4)

        mean_gen = Variable(xp.asarray(np.repeat(cuda.to_cpu(mean.data), gen_num, axis=0)), volatile=True)
        var_gen  = Variable(xp.asarray(np.repeat(cuda.to_cpu(var.data), gen_num, axis=0)), volatile=True)
        y = Variable(xp.asarray(np.repeat(cuda.to_cpu(y_gen.data), rec_num, axis=0)), volatile=True)
        rand = xp.random.normal(0, 1, var_gen.data.shape).astype(np.float32)
        z  = mean_gen + self.gy(y) + F.exp(var_gen) * Variable(rand, volatile=True)
        g1 = F.reshape(F.relu(self.gnorm1(self.g1(z), test=True)), (z.data.shape[0], 512, 6, 6))
        g2 = F.relu(self.gnorm2(self.g2(g1), test=True))
        g3 = F.relu(self.gnorm3(self.g3(g2), test=True))
        g4 = F.relu(self.gnorm4(self.g4(g3), test=True))
        g5 = F.sigmoid(self.g5(g4))
        return g5

class EtlVAENet2(chainer.Chain):
    def __init__(self):
        super(EtlVAENet2, self).__init__(
            r1   = L.Convolution2D(1, 64, 4, stride=2, pad=1),
            r2   = L.Convolution2D(64, 128, 4, stride=2, pad=1),
            r3   = L.Convolution2D(128, 256, 4, stride=2, pad=1),
            r4   = L.Convolution2D(256, 512, 4, stride=2, pad=1),
            mean = L.Linear(6 * 6 * 512, 100, wscale=0.02 * math.sqrt(6 * 6 * 512)),
            var  = L.Linear(6 * 6 * 512, 100, wscale=0.02 * math.sqrt(6 * 6 * 512)),
            gy   = L.EmbedID(CHARACTER_NUM, 100),
            g1   = L.Linear(100, 6 * 6 * 512, wscale=0.02 * math.sqrt(100)),
            g2   = L.Deconvolution2D(512, 256, 4, stride=2, pad=1),
            g3   = L.Deconvolution2D(256, 128, 4, stride=2, pad=1),
            g4   = L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            g5   = L.Deconvolution2D(64, 1, 4, stride=2, pad=1),
#            r1   = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
#            r2   = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
#            r3   = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
#            r4   = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
#            mean = L.Linear(6 * 6 * 512, 100, wscale=0.02 * math.sqrt(6 * 6 * 512)),
#            var  = L.Linear(6 * 6 * 512, 100, wscale=0.02 * math.sqrt(6 * 6 * 512)),
#            gy   = L.EmbedID(CHARACTER_NUM, 100),
#            g1   = L.Linear(100, 6 * 6 * 512, wscale=0.02 * math.sqrt(100)),
#            g2   = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 512)),
#            g3   = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
#            g4   = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
#            g5   = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
        )

    def __call__(self, (x, y), train=True):
        xp = cuda.get_array_module(x.data)
        r1 = F.leaky_relu(self.r1(x))
        r2 = F.leaky_relu(self.r2(r1))
        r3 = F.leaky_relu(self.r3(r2))
        r4 = F.leaky_relu(self.r4(r3))
        mean = self.mean(r4) - self.gy(y)
        var = 0.5 * self.var(r4)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + self.gy(y) + F.exp(var) * Variable(rand, volatile=not train)
        g1 = F.reshape(F.relu(self.g1(z)), (z.data.shape[0], 512, 6, 6))
        g2 = F.relu(self.g2(g1))
        g3 = F.relu(self.g3(g2))
        g4 = F.relu(self.g4(g3))
        g5 = F.sigmoid(self.g5(g4))
        return (g5, mean, var)

    def generate(self, x_rec, y_rec, y_gen):
        assert x_rec.data.shape[0] == y_rec.data.shape[0]
        rec_num = x_rec.data.shape[0]
        gen_num = y_gen.data.shape[0]
        xp = cuda.get_array_module(x_rec.data)
        r1 = F.leaky_relu(self.r1(x_rec))
        r2 = F.leaky_relu(self.r2(r1))
        r3 = F.leaky_relu(self.r3(r2))
        r4 = F.leaky_relu(self.r4(r3))
        mean = self.mean(r4) - self.gy(y_rec)
        var = 0.5 * self.var(r4)

        mean_gen = Variable(xp.asarray(np.repeat(cuda.to_cpu(mean.data), gen_num, axis=0)), volatile=True)
        var_gen  = Variable(xp.asarray(np.repeat(cuda.to_cpu(var.data), gen_num, axis=0)), volatile=True)
        y = Variable(xp.asarray(np.repeat(cuda.to_cpu(y_gen.data), rec_num, axis=0)), volatile=True)
        rand = xp.random.normal(0, 1, var_gen.data.shape).astype(np.float32)
        z  = mean_gen + self.gy(y) + F.exp(var_gen) * Variable(rand, volatile=True)
        g1 = F.reshape(F.relu(self.g1(z)), (z.data.shape[0], 512, 6, 6))
        g2 = F.relu(self.g2(g1))
        g3 = F.relu(self.g3(g2))
        g4 = F.relu(self.g4(g3))
        g5 = F.sigmoid(self.g5(g4))
        return g5

class EtlVAENet48(chainer.Chain):

    def __init__(self):
        super(EtlVAENet48, self).__init__(
            gy = L.EmbedID(CHARACTER_NUM, 500),
            r1 = L.Linear(2304, 1000, wscale=0.02 * math.sqrt(2304)),
            r2 = L.Linear(1000, 1000, wscale=0.02 * math.sqrt(1000)),
            r3 = L.Linear(1000, 1000, wscale=0.02 * math.sqrt(1000)),
            mean = L.Linear(1000, 500, wscale=0.02 * math.sqrt(1000)),
            var = L.Linear(1000, 500, wscale=0.02 * math.sqrt(1000)),
            g1 = L.Linear(500, 1000, wscale=0.02 * math.sqrt(500)),
            g2 = L.Linear(1000, 1000, wscale=0.02 * math.sqrt(1000)),
            g3 = L.Linear(1000, 1000, wscale=0.02 * math.sqrt(1000)),
            g4 = L.Linear(1000, 2304, wscale=0.02 * math.sqrt(1000))
        )

    def __call__(self, (x, y), train=True):
        xp = cuda.get_array_module(x.data)
        h1 = F.relu(self.r1(x))
        h2 = F.relu(self.r2(h1))
        h3 = F.relu(self.r3(h2))
        mean = self.mean(h3) - self.gy(y)
        var  = 0.5 * self.var(h3)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + self.gy(y) + F.exp(var) * Variable(rand, volatile=not train)
        g1 = F.relu(self.g1(z))
        g2 = F.relu(self.g2(g1))
        g3 = F.relu(self.g3(g2))
        g4 = self.g4(g3)
        return (g4, mean, var)

    def generate(self, x_rec, y_rec, y_gen):
        assert x_rec.data.shape[0] == y_rec.data.shape[0]
        rec_num = x_rec.data.shape[0]
        gen_num = y_gen.data.shape[0]
        xp = cuda.get_array_module(x_rec.data)
        h1 = F.relu(self.r1(x_rec))
        h2 = F.relu(self.r2(h1))
        h3 = F.relu(self.r3(h2))
        mean = self.mean(h3) - self.gy(y_gen)
        var  = 0.5 * self.var(h3)

        mean_gen = Variable(xp.asarray(np.repeat(cuda.to_cpu(mean.data), gen_num, axis=0)), volatile=True)
        var_gen  = Variable(xp.asarray(np.repeat(cuda.to_cpu(var.data), gen_num, axis=0)), volatile=True)
        y_gen = Variable(xp.asarray(np.repeat(cuda.to_cpu(y_gen.data), rec_num, axis=0)), volatile=True)
        rand = xp.random.normal(0, 1, var_gen.data.shape).astype(np.float32)
        z  = mean_gen + self.gy(y_gen) + F.exp(var_gen) * Variable(rand, volatile=True)
        g1 = F.relu(self.g1(z))
        g2 = F.relu(self.g2(g1))
        g3 = F.relu(self.g3(g2))
        g4 = F.sigmoid(self.g4(g3))
        return g4

class EtlVAENet48Conv(chainer.Chain):
    def __init__(self):
        super(EtlVAENet48Conv, self).__init__(
            r1     = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            rnorm1 = L.BatchNormalization(64),
            r2     = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            rnorm2 = L.BatchNormalization(128),
            r3     = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            rnorm3 = L.BatchNormalization(256),
            mean   = L.Linear(6 * 6 * 256, 100, wscale=0.02 * math.sqrt(6 * 6 * 256)),
            var    = L.Linear(6 * 6 * 256, 100, wscale=0.02 * math.sqrt(6 * 6 * 256)),
            gy     = L.EmbedID(CHARACTER_NUM, 100),
            g1     = L.Linear(100, 6 * 6 * 256, wscale=0.02 * math.sqrt(100)),
            gnorm1 = L.BatchNormalization(6 * 6 * 256),
            g2     = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            gnorm2 = L.BatchNormalization(128),
            g3     = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            gnorm3 = L.BatchNormalization(64),
            g4     = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
        )

    def __call__(self, (x, y), train=True):
        xp = cuda.get_array_module(x.data)
        r1 = F.leaky_relu(self.rnorm1(self.r1(x), test=not train))
        r2 = F.leaky_relu(self.rnorm2(self.r2(r1), test=not train))
        r3 = F.leaky_relu(self.rnorm3(self.r3(r2), test=not train))
        mean = self.mean(r3) - self.gy(y)
        var = 0.5 * self.var(r3)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + self.gy(y) + F.exp(var) * Variable(rand, volatile=not train)
        g1 = F.reshape(F.relu(self.gnorm1(self.g1(z), test=not train)), (z.data.shape[0], 256, 6, 6))
        g2 = F.relu(self.gnorm2(self.g2(g1), test=not train))
        g3 = F.relu(self.gnorm3(self.g3(g2), test=not train))
        g4 = self.g4(g3)
        return (g4, mean, var)

    def generate(self, x_rec, y_rec, y_gen):
        assert x_rec.data.shape[0] == y_rec.data.shape[0]
        rec_num = x_rec.data.shape[0]
        gen_num = y_gen.data.shape[0]
        xp = cuda.get_array_module(x_rec.data)
        r1 = F.leaky_relu(self.rnorm1(self.r1(x_rec), test=True))
        r2 = F.leaky_relu(self.rnorm2(self.r2(r1), test=True))
        r3 = F.leaky_relu(self.rnorm3(self.r3(r2), test=True))
        mean = self.mean(r3) - self.gy(y_rec)
        var = 0.5 * self.var(r3)

        mean_gen = Variable(xp.asarray(np.repeat(cuda.to_cpu(mean.data), gen_num, axis=0)), volatile=True)
        var_gen  = Variable(xp.asarray(np.repeat(cuda.to_cpu(var.data), gen_num, axis=0)), volatile=True)
        y = Variable(xp.asarray(np.repeat(cuda.to_cpu(y_gen.data), rec_num, axis=0)), volatile=True)
        rand = xp.random.normal(0, 1, var_gen.data.shape).astype(np.float32)
        z  = mean_gen + self.gy(y) + F.exp(var_gen) * Variable(rand, volatile=True)
        g1 = F.reshape(F.relu(self.gnorm1(self.g1(z), test=True)), (z.data.shape[0], 256, 6, 6))
        g2 = F.relu(self.gnorm2(self.g2(g1), test=True))
        g3 = F.relu(self.gnorm3(self.g3(g2), test=True))
        g4 = self.g4(g3)
        return g4

class EtlVAENet48Conv2(chainer.Chain):
    def __init__(self):
        super(EtlVAENet48Conv2, self).__init__(
            r1     = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            rnorm1 = L.BatchNormalization(64),
            r2     = ConvPool(64),
            r3     = ConvPool(128),
            mean   = L.Linear(6 * 6 * 256, 100, wscale=0.02 * math.sqrt(6 * 6 * 256)),
            var    = L.Linear(6 * 6 * 256, 100, wscale=0.02 * math.sqrt(6 * 6 * 256)),
            gy     = L.EmbedID(CHARACTER_NUM, 100),
            g1     = L.Linear(100, 6 * 6 * 256, wscale=0.02 * math.sqrt(100)),
            gnorm1 = L.BatchNormalization(6 * 6 * 256),
            g2     = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            gnorm2 = L.BatchNormalization(128),
            g3     = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            gnorm3 = L.BatchNormalization(64),
            g4     = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
        )

    def __call__(self, (x, y), train=True):
        xp = cuda.get_array_module(x.data)
        r1 = F.leaky_relu(self.rnorm1(self.r1(x), test=not train))
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        mean = self.mean(r3) - self.gy(y)
        var = 0.5 * self.var(r3)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + self.gy(y) + F.exp(var) * Variable(rand, volatile=not train)
        g1 = F.reshape(F.relu(self.gnorm1(self.g1(z), test=not train)), (z.data.shape[0], 256, 6, 6))
        g2 = F.relu(self.gnorm2(self.g2(g1), test=not train))
        g3 = F.relu(self.gnorm3(self.g3(g2), test=not train))
        g4 = self.g4(g3)
        return (g4, mean, var)

    def generate(self, x_rec, y_rec, y_gen):
        assert x_rec.data.shape[0] == y_rec.data.shape[0]
        rec_num = x_rec.data.shape[0]
        gen_num = y_gen.data.shape[0]
        xp = cuda.get_array_module(x_rec.data)
        r1 = F.leaky_relu(self.rnorm1(self.r1(x_rec), test=True))
        r2 = self.r2(r1, train=False)
        r3 = self.r3(r2, train=False)
        mean = self.mean(r3) - self.gy(y_rec)
        var = 0.5 * self.var(r3)

        mean_gen = Variable(xp.asarray(np.repeat(cuda.to_cpu(mean.data), gen_num, axis=0)), volatile=True)
        var_gen  = Variable(xp.asarray(np.repeat(cuda.to_cpu(var.data), gen_num, axis=0)), volatile=True)
        y = Variable(xp.asarray(np.repeat(cuda.to_cpu(y_gen.data), rec_num, axis=0)), volatile=True)
        rand = xp.random.normal(0, 1, var_gen.data.shape).astype(np.float32)
        z  = mean_gen + self.gy(y) + F.exp(var_gen) * Variable(rand, volatile=True)
        g1 = F.reshape(F.relu(self.gnorm1(self.g1(z), test=True)), (z.data.shape[0], 256, 6, 6))
        g2 = F.relu(self.gnorm2(self.g2(g1), test=True))
        g3 = F.relu(self.gnorm3(self.g3(g2), test=True))
        g4 = self.g4(g3)
        return g4

class Generator(chainer.Chain):
    def __init__(self, latent_size=100):
        super(Generator, self).__init__(
            g1    = L.Linear(latent_size, 6 * 6 * 512, wscale=0.02 * math.sqrt(latent_size)),
            norm1 = L.BatchNormalization(6 * 6 * 512),
            g2    = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 512)),
            norm2 = L.BatchNormalization(256),
            g3    = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm3 = L.BatchNormalization(128),
            g4    = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm4 = L.BatchNormalization(64),
            g5    = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
        )

    def __call__(self, z, train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)), (z.data.shape[0], 512, 6, 6))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3), test=not train))
        return F.sigmoid(self.g5(h4))

class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            dc1   = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(64),
            dc2   = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            norm2 = L.BatchNormalization(128),
            dc3   = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm3 = L.BatchNormalization(256),
            dc4   = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm4 = L.BatchNormalization(512),
            dc5   = L.Linear(6 * 6 * 512, 2, wscale=0.02 * math.sqrt(6 * 6 * 512)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        return self.dc5(h4)

class Generator2(chainer.Chain):
    def __init__(self):
        super(Generator2, self).__init__(
            gy    = L.EmbedID(CHARACTER_NUM, 100),
            g1    = L.Linear(100, 6 * 6 * 512, wscale=0.02 * math.sqrt(100)),
            norm1 = L.BatchNormalization(6 * 6 * 512),
            g2    = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 512)),
            norm2 = L.BatchNormalization(256),
            g3    = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm3 = L.BatchNormalization(128),
            g4    = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm4 = L.BatchNormalization(64),
            g5    = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
        )

    def __call__(self, (z, y), train=True):
        h0 = z + self.gy(y)
        h1 = F.reshape(F.relu(self.norm1(self.g1(h0), test=not train)), (z.data.shape[0], 512, 6, 6))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3), test=not train))
        return F.sigmoid(self.g5(h4))

class Discriminator2(chainer.Chain):
    def __init__(self):
        super(Discriminator2, self).__init__(
            dc1   = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(64),
            dc2   = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            norm2 = L.BatchNormalization(128),
            dc3   = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm3 = L.BatchNormalization(256),
            dc4   = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm4 = L.BatchNormalization(512),
            dc5   = L.Linear(6 * 6 * 512, 100, wscale=0.02 * math.sqrt(6 * 6 * 512)),
            norm5 = L.BatchNormalization(100),
            dc6   = L.Linear(100, CHARACTER_NUM + 1, wscale=0.02 * math.sqrt(100)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        h5 = F.leaky_relu(self.norm5(self.dc5(h4), test=not train))
        return self.dc6(h5)

class Generator3(chainer.Chain):
    def __init__(self):
        super(Generator3, self).__init__(
            gy    = L.EmbedID(CHARACTER_NUM, 100),
            g1    = L.Linear(100, 6 * 6 * 256, wscale=0.02 * math.sqrt(100)),
            norm1 = L.BatchNormalization(6 * 6 * 256),
            g2    = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm2 = L.BatchNormalization(128),
            g3    = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm3 = L.BatchNormalization(64),
            g4    = L.Deconvolution2D(64, 32, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            norm4 = L.BatchNormalization(32),
            g5    = L.Deconvolution2D(32, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32)),
        )

    def __call__(self, (z, y), train=True):
        h0 = z + self.gy(y)
        h1 = F.reshape(F.relu(self.norm1(self.g1(h0), test=not train)), (z.data.shape[0], 256, 6, 6))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3), test=not train))
        return F.sigmoid(self.g5(h4))

class Discriminator3(chainer.Chain):
    def __init__(self):
        super(Discriminator3, self).__init__(
            dc1   = L.Convolution2D(1, 32, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(32),
            dc2   = L.Convolution2D(32, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32)),
            norm2 = L.BatchNormalization(64),
            dc3   = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            norm3 = L.BatchNormalization(128),
            dc4   = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm4 = L.BatchNormalization(256),
            dc5   = L.Linear(6 * 6 * 256, 100, wscale=0.02 * math.sqrt(6 * 6 * 256)),
            norm5 = L.BatchNormalization(100),
            dc6   = L.Linear(100, CHARACTER_NUM + 1, wscale=0.02 * math.sqrt(100)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        h5 = F.leaky_relu(self.norm5(self.dc5(h4), test=not train))
        return self.dc6(h5)

class Generator4(chainer.Chain):
    def __init__(self):
        latent_size = 100
        super(Generator4, self).__init__(
            gy    = L.EmbedID(CHARACTER_NUM, latent_size),
            g1    = L.Linear(latent_size, 6 * 6 * 512, wscale=0.02 * math.sqrt(latent_size)),
            norm1 = L.BatchNormalization(6 * 6 * 512),
            g2    = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 512)),
            norm2 = L.BatchNormalization(256),
            g3    = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm3 = L.BatchNormalization(128),
            g4    = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm4 = L.BatchNormalization(64),
            g5    = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
        )

    def __call__(self, (z, y), train=True):
        h0 = z + self.gy(y)
        h1 = F.reshape(F.relu(self.norm1(self.g1(h0), test=not train)), (z.data.shape[0], 512, 6, 6))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3), test=not train))
        return F.sigmoid(self.g5(h4))

class Discriminator4(chainer.Chain):
    def __init__(self):
        latent_size = 100
        super(Discriminator4, self).__init__(
            dc1   = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(64),
            dc2   = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            norm2 = L.BatchNormalization(128),
            dc3   = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm3 = L.BatchNormalization(256),
            dc4   = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm4 = L.BatchNormalization(512),
            dc5   = L.Linear(6 * 6 * 512, latent_size, wscale=0.02 * math.sqrt(6 * 6 * 512)),
            norm5 = L.BatchNormalization(latent_size),
            dc6   = L.Linear(6 * 6 * 512, 2, wscale=0.02 * math.sqrt(6 * 6 * 512)),
            dc7   = L.Linear(latent_size, CHARACTER_NUM, wscale=0.02 * math.sqrt(latent_size)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        h5 = F.leaky_relu(self.norm5(self.dc5(h4), test=not train))
        return (self.dc6(h4), self.dc7(h5))

class Generator5(chainer.Chain):
    def __init__(self):
        LATENT_SIZE = 100
        super(Generator5, self).__init__(
            gy    = L.EmbedID(CHARACTER_NUM, LATENT_SIZE),
            g1    = L.Linear(LATENT_SIZE, 6 * 6 * 512, wscale=0.02 * math.sqrt(LATENT_SIZE)),
            norm1 = L.BatchNormalization(6 * 6 * 512),
            g2    = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 512)),
            norm2 = L.BatchNormalization(256),
            g3    = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm3 = L.BatchNormalization(128),
            g4    = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm4 = L.BatchNormalization(64),
            g5    = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
        )

    def __call__(self, (z, y), train=True):
        h0 = z + self.gy(y)
        h1 = F.reshape(F.relu(self.norm1(self.g1(h0), test=not train)), (z.data.shape[0], 512, 6, 6))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3), test=not train))
        return F.sigmoid(self.g5(h4))

class Discriminator5(chainer.Chain):
    def __init__(self):
        LATENT_SIZE = 100
        super(Discriminator5, self).__init__(
            dc1   = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(64),
            dc2   = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            norm2 = L.BatchNormalization(128),
            dc3   = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm3 = L.BatchNormalization(256),
            dc4   = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm4 = L.BatchNormalization(512),
            dc5   = L.Linear(6 * 6 * 512, LATENT_SIZE, wscale=0.02 * math.sqrt(6 * 6 * 512)),
            norm5 = L.BatchNormalization(LATENT_SIZE),
            dc6   = L.Linear(LATENT_SIZE, 2, wscale=0.02 * math.sqrt(LATENT_SIZE)),
            dcy   = L.EmbedID(CHARACTER_NUM, LATENT_SIZE),
        )
        self.dcy.W.data *= 0.02 * math.sqrt(CHARACTER_NUM)

    def __call__(self, (x, y), train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        h5 = F.leaky_relu(self.norm5(self.dc5(h4), test=not train))
        return self.dc6(h5 - self.dcy(y))

class Generator6(chainer.Chain):
    def __init__(self, latent_size=100):
        size = 8
        super(Generator6, self).__init__(
            g1    = L.Linear(latent_size, 4 * 4 * 24 * size, wscale=0.02 * math.sqrt(latent_size)),
            norm1 = L.BatchNormalization(4 * 4 * 24 * size),
            g2    = L.Deconvolution2D(24 * size, 16 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 24 * size)),
            norm2 = L.BatchNormalization(16 * size),
            g3    = L.Deconvolution2D(16 * size, 12 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 16 * size)),
            norm3 = L.BatchNormalization(12 * size),
            g4    = L.Deconvolution2D(12 * size, 8 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 12 * size)),
            norm4 = L.BatchNormalization(8 * size),
            g5    = L.Deconvolution2D(8 * size, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 8 * size)),
        )
        self.size = size

    def __call__(self, z, train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)), (z.data.shape[0], 64 * self.size, 4, 4))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3), test=not train))
#        return F.sigmoid(self.g5(h4))
        return self.g5(h4)

class Discriminator6(chainer.Chain):
    def __init__(self):
        size = 8
        super(Discriminator6, self).__init__(
            dc1   = L.Convolution2D(1, 8 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(8 * size),
            dc2   = L.Convolution2D(8 * size, 12 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 8 * size)),
            norm2 = L.BatchNormalization(12 * size),
            dc3   = L.Convolution2D(12 * size, 16 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 12 * size)),
            norm3 = L.BatchNormalization(16 * size),
            dc4   = L.Convolution2D(16 * size, 24 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 16 * size)),
            norm4 = L.BatchNormalization(24 * size),
            dc5   = L.Linear(4 * 4 * 24 * size, 2, wscale=0.02 * math.sqrt(4 * 4 * 24 * size)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        return self.dc5(h4)

class Generator7(chainer.Chain):
    def __init__(self, latent_size=100):
        size = 8
        super(Generator7, self).__init__(
            g1    = L.Linear(latent_size, 4 * 4 * 64 * size, wscale=0.02 * math.sqrt(latent_size)),
            norm1 = L.BatchNormalization(4 * 4 * 64 * size),
            g2    = L.Deconvolution2D(64 * size, 32 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64 * size)),
            norm2 = L.BatchNormalization(32 * size),
            g3    = L.Deconvolution2D(32 * size, 16 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32 * size)),
            norm3 = L.BatchNormalization(16 * size),
            g4    = L.Deconvolution2D(16 * size, 8 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 16 * size)),
            norm4 = L.BatchNormalization(8 * size),
            g5    = L.Deconvolution2D(8 * size, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 8 * size)),
        )
        self.size = size

    def __call__(self, z, train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)), (z.data.shape[0], 64 * self.size, 4, 4))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3), test=not train))
#        return F.sigmoid(self.g5(h4))
        return self.g5(h4)

class Discriminator7(chainer.Chain):
    def __init__(self):
        size = 8
        super(Discriminator7, self).__init__(
            dc1   = L.Convolution2D(1, 8 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(8 * size),
            dc2   = L.Convolution2D(8 * size, 16 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 8 * size)),
            norm2 = L.BatchNormalization(16 * size),
            dc3   = L.Convolution2D(16 * size, 32 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 16 * size)),
            norm3 = L.BatchNormalization(32 * size),
            dc4   = L.Convolution2D(32 * size, 64 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32 * size)),
            norm4 = L.BatchNormalization(64 * size),
            dc5   = L.Linear(4 * 4 * 64 * size, 2, wscale=0.02 * math.sqrt(4 * 4 * 64 * size)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        return self.dc5(h4)

class Generator8(chainer.Chain):
    def __init__(self, latent_size=100):
        size = 8
        super(Generator8, self).__init__(
            g1    = L.Linear(latent_size, 8 * 8 * 32 * size, wscale=0.02 * math.sqrt(latent_size)),
            norm1 = L.BatchNormalization(8 * 8 * 32 * size),
            g2    = L.Deconvolution2D(32 * size, 16 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32 * size)),
            norm2 = L.BatchNormalization(16 * size),
            g3    = L.Deconvolution2D(16 * size, 8 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 16 * size)),
            norm3 = L.BatchNormalization(8 * size),
            g4    = L.Deconvolution2D(8 * size, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 8 * size)),
        )
        self.size = size

    def __call__(self, z, train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)), (z.data.shape[0], 32 * self.size, 8, 8))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        return self.g4(h3)

class Discriminator8(chainer.Chain):
    def __init__(self):
        size = 8
        super(Discriminator8, self).__init__(
            dc1   = L.Convolution2D(1, 8 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(8 * size),
            dc2   = L.Convolution2D(8 * size, 16 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 8 * size)),
            norm2 = L.BatchNormalization(16 * size),
            dc3   = L.Convolution2D(16 * size, 32 * size, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 16 * size)),
            norm3 = L.BatchNormalization(32 * size),
            dc4   = L.Linear(8 * 8 * 32 * size, 2, wscale=0.02 * math.sqrt(8 * 8 * 32 * size)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        return self.dc4(h3)

class Generator9(chainer.Chain):
    def __init__(self, latent_size=100):
        size = 1
        super(Generator9, self).__init__(
            g1    = L.Linear(latent_size, 12 * 12 * 32 * size, wscale=0.02 * math.sqrt(latent_size)),
            norm1 = L.BatchNormalization(12 * 12 * 32 * size),
            g2    = L.Deconvolution2D(32 * size, 16 * size, 6, stride=2, pad=2, wscale=0.02 * math.sqrt(6 * 6 * 32 * size)),
            norm2 = L.BatchNormalization(16 * size),
            g3    = L.Deconvolution2D(16 * size, 8 * size, 6, stride=2, pad=2, wscale=0.02 * math.sqrt(6 * 6 * 16 * size)),
            norm3 = L.BatchNormalization(8 * size),
            g4    = L.Deconvolution2D(8 * size, 1, 6, stride=2, pad=2, wscale=0.02 * math.sqrt(6 * 6 * 8 * size)),
        )
        self.size = size

    def __call__(self, z, train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)), (z.data.shape[0], 32 * self.size, 12, 12))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        return self.g4(h3)

class Discriminator9(chainer.Chain):
    def __init__(self):
        size = 1
        super(Discriminator9, self).__init__(
            dc1   = L.Convolution2D(1, 8 * size, 6, stride=2, pad=2, wscale=0.02 * math.sqrt(6 * 6)),
            norm1 = L.BatchNormalization(8 * size),
            dc2   = L.Convolution2D(8 * size, 16 * size, 6, stride=2, pad=2, wscale=0.02 * math.sqrt(6 * 6 * 8 * size)),
            norm2 = L.BatchNormalization(16 * size),
            dc3   = L.Convolution2D(16 * size, 32 * size, 6, stride=2, pad=2, wscale=0.02 * math.sqrt(6 * 6 * 16 * size)),
            norm3 = L.BatchNormalization(32 * size),
            dc4   = L.Linear(12 * 12 * 32 * size, 2, wscale=0.02 * math.sqrt(12 * 12 * 32 * size)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        return self.dc4(h3)

class Generator10(chainer.Chain):
    def __init__(self, latent_size=100):
        size = 1
        super(Generator10, self).__init__(
            g1    = L.Linear(latent_size, 12 * 12 * 16 * size, wscale=0.02 * math.sqrt(latent_size)),
            norm1 = L.BatchNormalization(12 * 12 * 16 * size),
            g2    = L.Deconvolution2D(16 * size, 12 * size, 6, stride=2, pad=2, wscale=0.02 * math.sqrt(6 * 6 * 16 * size)),
            norm2 = L.BatchNormalization(12 * size),
            g3    = L.Deconvolution2D(12 * size, 8 * size, 6, stride=2, pad=2, wscale=0.02 * math.sqrt(6 * 6 * 12 * size)),
            norm3 = L.BatchNormalization(8 * size),
            g4    = L.Deconvolution2D(8 * size, 1, 6, stride=2, pad=2, wscale=0.02 * math.sqrt(6 * 6 * 8 * size)),
        )
        self.size = size

    def __call__(self, z, train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)), (z.data.shape[0], 16 * self.size, 12, 12))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        return self.g4(h3)

class Discriminator10(chainer.Chain):
    def __init__(self):
        size = 1
        super(Discriminator10, self).__init__(
            dc1   = L.Convolution2D(1, 8 * size, 6, stride=2, pad=2, wscale=0.02 * math.sqrt(6 * 6)),
            norm1 = L.BatchNormalization(8 * size),
            dc2   = L.Convolution2D(8 * size, 12 * size, 6, stride=2, pad=2, wscale=0.02 * math.sqrt(6 * 6 * 8 * size)),
            norm2 = L.BatchNormalization(12 * size),
            dc3   = L.Convolution2D(12 * size, 16 * size, 6, stride=2, pad=2, wscale=0.02 * math.sqrt(6 * 6 * 12 * size)),
            norm3 = L.BatchNormalization(16 * size),
            dc4   = L.Linear(12 * 12 * 16 * size, 2, wscale=0.02 * math.sqrt(12 * 12 * 16 * size)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        return self.dc4(h3)

class Generator11(chainer.Chain):
    def __init__(self):
        LATENT_SIZE = 100
        super(Generator11, self).__init__(
            gy    = L.EmbedID(CHARACTER_NUM, LATENT_SIZE),
            g1    = L.Linear(LATENT_SIZE, 6 * 6 * 256, wscale=0.02 * math.sqrt(LATENT_SIZE)),
            norm1 = L.BatchNormalization(6 * 6 * 256),
            g2    = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm2 = L.BatchNormalization(128),
            g3    = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm3 = L.BatchNormalization(64),
            g4    = L.Deconvolution2D(64, 32, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            norm4 = L.BatchNormalization(32),
            g5    = L.Deconvolution2D(32, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32)),
        )
        self.gy.W.data *= 0.02 * math.sqrt(CHARACTER_NUM)

    def __call__(self, (z, y), train=True):
        h0 = z + self.gy(y)
        h1 = F.reshape(F.relu(self.norm1(self.g1(h0), test=not train)), (z.data.shape[0], 256, 6, 6))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3), test=not train))
        return F.sigmoid(self.g5(h4))

class Discriminator11(chainer.Chain):
    def __init__(self):
        LATENT_SIZE = 100
        super(Discriminator11, self).__init__(
            dc1   = L.Convolution2D(1, 32, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(32),
            dc2   = L.Convolution2D(32, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32)),
            norm2 = L.BatchNormalization(64),
            dc3   = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            norm3 = L.BatchNormalization(128),
            dc4   = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm4 = L.BatchNormalization(256),
            dc5   = L.Convolution2D(256, 32, 1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm5 = L.BatchNormalization(32),
            dcy   = L.EmbedID(CHARACTER_NUM, 6 * 6 * 32),
            dc6   = L.Linear(6 * 6 * 32, 2, wscale=0.02 * math.sqrt(6 * 6 * 32)),
        )
        self.dcy.W.data *= 0.02 * math.sqrt(CHARACTER_NUM)

    def __call__(self, (x, y), train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        h5 = F.leaky_relu(self.norm5(F.reshape(self.dc5(h4), (x.data.shape[0], 6 * 6 * 32,)) + self.dcy(y), test=not train))
        return self.dc6(h5)

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

class Generator48_2(chainer.Chain):
    def __init__(self):
        LATENT_SIZE = 100
        super(Generator48_2, self).__init__(
            gy    = L.EmbedID(CHARACTER_NUM, 100),
            g1    = L.Linear(100, 6 * 6 * 256, wscale=0.02 * math.sqrt(100)),
            norm1 = L.BatchNormalization(6 * 6 * 256),
            g2    = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm2 = L.BatchNormalization(128),
            g3    = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm3 = L.BatchNormalization(64),
            g4    = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
        )
#        self.gy.W.data *= 0.02 * math.sqrt(CHARACTER_NUM)

    def __call__(self, (z, y), train=True):
        h0 = z + self.gy(y)
        h1 = F.reshape(F.relu(self.norm1(self.g1(h0), test=not train)), (z.data.shape[0], 256, 6, 6))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        return F.sigmoid(self.g4(h3))

class Discriminator48_2(chainer.Chain):
    def __init__(self):
        super(Discriminator48_2, self).__init__(
            dc1   = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(64),
            dc2   = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            norm2 = L.BatchNormalization(128),
            dc3   = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm3 = L.BatchNormalization(256),
            dc4   = L.Linear(6 * 6 * 256, 2, wscale=0.02 * math.sqrt(6 * 6 * 256)),
            dcy   = L.EmbedID(CHARACTER_NUM, 6 * 6 * 256),
        )
#        self.dcy.W.data *= 0.02 * math.sqrt(CHARACTER_NUM)

    def __call__(self, (x, y), train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(F.reshape(self.norm3(self.dc3(h2), test=not train), (x.data.shape[0], 6 * 6 * 256)) + self.dcy(y))
        return self.dc4(h3)

class Generator48_3(chainer.Chain):
    def __init__(self):
        super(Generator48_3, self).__init__(
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
        return self.g4(h3)

class Discriminator48_3(chainer.Chain):
    def __init__(self):
        super(Discriminator48_3, self).__init__(
            dc1   = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(64),
            dc2   = ConvPool(64),
            dc3   = ConvPool(128),
            dc4   = L.Linear(6 * 6 * 256, 100, wscale=0.02 * math.sqrt(6 * 6 * 256)),
            norm4 = L.BatchNormalization(100),
            dc5   = L.Linear(100, CHARACTER_NUM + 1, wscale=0.02 * math.sqrt(100)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = self.dc2(h1)
        h3 = self.dc3(h2)
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        return self.dc5(h4)

class Generator_(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            g1    = L.Linear(200, 6 * 6 * 160, wscale=0.02 * math.sqrt(200)),
            norm1 = L.BatchNormalization(6 * 6 * 160),
            g2    = L.Deconvolution2D(160, 80, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 160)),
            norm2 = L.BatchNormalization(80),
            g3    = L.Deconvolution2D(80, 40, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 80)),
            norm3 = L.BatchNormalization(40),
            g4    = L.Deconvolution2D(40, 20, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 40)),
            norm4 = L.BatchNormalization(20),
            g5    = L.Deconvolution2D(20, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 20)),
        )

    def __call__(self, z, train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)), (z.data.shape[0], 160, 6, 6))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3), test=not train))
        return F.sigmoid(self.g5(h4))

class Discriminator_(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            dc1   = L.Convolution2D(1, 20, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(20)),
            norm1 = L.BatchNormalization(20),
            dc2   = L.Convolution2D(20, 40, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 20)),
            norm2 = L.BatchNormalization(40),
            dc3   = L.Convolution2D(40, 80, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 40)),
            norm3 = L.BatchNormalization(80),
            dc4   = L.Convolution2D(80, 160, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 80)),
            norm4 = L.BatchNormalization(160),
            dc5   = L.Linear(6 * 6 * 160, 2, wscale=0.02 * math.sqrt(6 * 6 * 160)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        return self.dc5(h4)
