from chainer import link
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy as sce
from chainer.functions.loss import mean_squared_error as mse
from chainer.functions.loss import vae

class Model(link.Chain):
    def __init__(self, predictor):
        super(Model, self).__init__(predictor=predictor)
        self.y = None
        self.loss = None
        self.accuracy = None
        self.compute_accuracy = True

    def __call__(self, x, t, train=True):
        self.y = self.predictor(x, train=train)
        self.loss = self.calc_loss(self.y, t)
        if self.compute_accuracy:
            self.accuracy = self.calc_accuracy(self.y, t)
        return self.loss

    def calc_loss(self, y, t):
        raise NotImplementedError

    def calc_accuracy(self, y, t):
        raise NotImplementedError

class Classifier(Model):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor)

    def calc_loss(self, y, t):
        return sce.softmax_cross_entropy(y, t)

    def calc_accuracy(self, y, t):
        return accuracy.accuracy(y, t)

class VAEModel(Model):
    def __init__(self, predictor):
        super(VAEModel, self).__init__(predictor)

    def calc_loss(self, (y, mean, var), t):
        return mse.mean_squared_error(y, t) + vae.gaussian_kl_divergence(mean, var) / float(y.data.size)

    def calc_accuracy(self, (y, mean, var), t):
        return mse.mean_squared_error(y, t)
