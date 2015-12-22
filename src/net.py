import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

WIDTH = 128
HEIGHT = 127
IN_SIZE = WIDTH * HEIGHT
OUT_SIZE = 3036

class EtlNet(chainer.Chain):
    def __init__(self):
        super(EtlNet, self).__init__(
            l1 = L.Linear(IN_SIZE, 4096),
            l2 = L.Linear(4096, 4096),
            l3 = L.Linear(4096, OUT_SIZE)
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
            l6 = L.Linear(4096, OUT_SIZE)
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
            rec1_x = L.Linear(IN_SIZE, 4096),
            rec1_y = L.EmbedID(OUT_SIZE, 4096),
            rec2 = L.Linear(4096, 4096),
            rec_mean = L.Linear(4096, 300),
            rec_var  = L.Linear(4096, 300),
            gen1_z = L.Linear(300, 4096),
            gen1_y = L.EmbedID(OUT_SIZE, 4096),
            gen2 = L.Linear(4096, 4096),
            gen3 = L.Linear(4096, IN_SIZE)
        )

    def __call__(self, (x_var, y_var), train=True):
        xp = cuda.get_array_module(x_var.data)
        h1 = F.relu(self.rec1_x(x_var) + self.rec1_y(y_var))
        h2 = F.relu(self.rec2(h1))
        mean = self.rec_mean(h2)
        var  = 0.5 * self.rec_var(h2)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + F.exp(var) * Variable(rand, volatile=not train)
        g1 = F.relu(self.gen1_z(z) + self.gen1_y(y_var))
        g2 = F.relu(self.gen2(g1))
        g3 = F.sigmoid(self.gen3(g2))
        return (g3, mean, var)

    def generate(self, x_rec, y_rec, y_gen):
        assert x_rec.data.shape[0] == y_rec.data.shape[0]
        rec_num = x_rec.data.shape[0]
        gen_num = y_gen.data.shape[0]
        xp = cuda.get_array_module(x_rec.data)
        h1 = F.relu(self.rec1_x(x_rec) + self.rec1_y(y_rec))
        h2 = F.relu(self.rec2(h1))
        mean = self.rec_mean(h2)
        var  = 0.5 * self.rec_var(h2)

        mean_gen = Variable(xp.asarray(np.repeat(cuda.to_cpu(mean.data), gen_num, axis=0)), volatile=True)
        var_gen  = Variable(xp.asarray(np.repeat(cuda.to_cpu(var.data), gen_num, axis=0)), volatile=True)
        y_gen = Variable(xp.asarray(np.repeat(cuda.to_cpu(y_gen.data), rec_num, axis=0)), volatile=True)
        rand = xp.random.normal(0, 1, var_gen.data.shape).astype(np.float32)
        z  = mean_gen + F.exp(var_gen) * Variable(rand, volatile=True)
        g1 = F.relu(self.gen1_z(z) + self.gen1_y(y_gen))
        g2 = F.relu(self.gen2(g1))
        g3 = F.sigmoid(self.gen3(g2))
        return g3

class EtlVAE1Net(chainer.Chain):
    def __init__(self):
        super(EtlVAE1Net, self).__init__(
            rec1_x = L.Linear(IN_SIZE, 4000),
            rec2 = L.Linear(4000, 4000),
            rec3 = L.Linear(4000, 4000),
            rec_mean = L.Linear(4000, 300),
            rec_var  = L.Linear(4000, 300),
            gen1_z = L.Linear(300, 4000),
            gen2 = L.Linear(4000, 4000),
            gen3 = L.Linear(4000, 4000),
            gen4 = L.Linear(4000, IN_SIZE)
        )

    def __call__(self, (x, y), train=True):
        xp = cuda.get_array_module(x.data)
        h1 = F.relu(self.rec1_x(x))
        h2 = F.relu(self.rec2(h1))
        h3 = F.relu(self.rec3(h2))
        mean = self.rec_mean(h3)
        var  = 0.5 * self.rec_var(h2)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + F.exp(var) * Variable(rand, volatile=not train)
        g1 = F.relu(self.gen1_z(z))
        g2 = F.relu(self.gen2(g1))
        g3 = F.relu(self.gen3(g2))
        g = F.sigmoid(self.gen4(g3))
        return (g, mean, var)

    def generate(self, gen_num):
        z  = xp.random.normal(0, 1, (gen_num, 300)).astype(np.float32)
        g1 = F.relu(self.gen1_z(z))
        g2 = F.relu(self.gen2(g1))
        g3 = F.sigmoid(self.gen3(g2))
        return g3
