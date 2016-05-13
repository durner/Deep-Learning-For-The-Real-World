import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
sys.path.append(current_dir + '..')

import climin
import climin.util
import numpy
import theano
import theano.tensor as tensor
import utils
import matplotlib.pyplot as pyplot

class AutoEncoder(object):

    def __init__(self, x, numpy_rng, n_visible, n_hidden, lambda_param=0.1, sparse_cost=False,
                 W=None, bhid=None, bvis=None, kl_cost=False, activation=tensor.nnet.sigmoid):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.activation = activation
        self.x = x
        self.lambda_param = lambda_param

        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W')

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
            )


        self.weights = W
        self.b_hidden = bhid

        self.b_visible = bvis
        self.weights_prime = self.weights.T

        self.template = [(n_visible, n_hidden), n_hidden, n_visible]
        self.params = [self.weights, self.b_hidden, self.b_visible]

        if sparse_cost:
            print "sparse"
            self.cost = self.get_sparse_cost()
        elif kl_cost:
            print "kl"
            self.cost = self.get_kl_cost()
        else:
            self.cost = self.get_cost()

        self.gparams = [tensor.grad(self.cost, param) for param in self.params]

        self.loss_gradient = theano.function(
            inputs=[self.x],
            outputs=self.gparams
        )
        self.loss_overall = theano.function(
            inputs=[self.x],
            outputs=self.cost
        )


    def get_hidden_values(self, input):
        return self.activation(tensor.dot(input, self.weights) + self.b_hidden)

    def get_reconstructed_input(self, hidden):
        return self.activation(tensor.dot(hidden, self.weights_prime) + self.b_visible)

    def set_params(self, params):
        [w, bh, bv] = climin.util.shaped_from_flat(params, self.template)
        self.weights.set_value(w)
        self.b_hidden.set_value(bh)
        self.b_visible.set_value(bv)

    def get_params(self):
        flat, (w, bh, bv) = climin.util.empty_with_views(self.template)
        return numpy.concatenate([self.weights.get_value().flatten(),
                                  self.b_hidden.get_value().flatten(),
                                  self.b_visible.get_value().flatten()])

    def loss_grad(self, params, input):
        self.set_params(params)
        return numpy.concatenate([grad.flatten() for grad in self.loss_gradient(input)])

    def loss_func(self, params, input):
        self.set_params(params)
        return self.loss_overall(input)

    def get_updates(self, learning_rate):
        return [(param, param - learning_rate * gparam)for param, gparam in
                zip(self.params, tensor.grad(self.cost, self.params))]

    def get_cost(self):
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)

        # squared error

        cost = tensor.mean(tensor.sum((z - self.x)**2, axis=1))
        # cost = tensor.mean(- tensor.sum(self.x * tensor.log(z) + (1 - self.x) * tensor.log(1 - z), axis=1))

        return cost

    def get_sparse_cost(self):
        y = self.get_hidden_values(self.x)
        sparse_cost = self.get_cost() + tensor.mean(self.lambda_param * tensor.sum(tensor.abs_(y), axis=1))
        return sparse_cost

    def get_kl_cost(self):
        y = self.get_hidden_values(self.x)
        p = self.lambda_param
        pprime = tensor.mean(y, axis=0)
        sparse_cost = p * tensor.log(p) - tensor.log(pprime) + (1 - p) * tensor.log(1 - p) - (1 - p) * tensor.log(1 - pprime)
        sparse_cost = tensor.mean(sparse_cost)
        sparse_cost += self.get_cost()
        return sparse_cost

def train(data, epochs=10, activation=tensor.nnet.sigmoid, name="errors.png",
          sparsity=True, lambda_param=0.1, nhidden=100, kl=False):
    x = tensor.matrix('x')
    index = tensor.lscalar()
    rng = numpy.random.RandomState(1337)
    train_set = data
    n_hidden = nhidden
    auto_encoder = AutoEncoder(
        x=x,
        numpy_rng=rng,
        n_visible=train_set.get_value().shape[1],
        n_hidden=n_hidden,
        lambda_param=lambda_param,
        sparse_cost=sparsity,
        kl_cost=kl
    )

    learning_rate = 0.1

    updates = auto_encoder.get_updates(learning_rate=learning_rate)

    batch_size = 200
    n_train_batches = train_set.get_value().shape[0] // batch_size

    train_da = theano.function(
        [index],
        auto_encoder.cost,
        updates=updates,
        givens={
            x: train_set[index * batch_size: (index + 1) * batch_size]
        }
    )

    for epoch in range(epochs):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print 'Current Epoch: ', epoch, '\t Current Cost: ', numpy.mean(c)

    name_rec = "autoencoderrec_" + str(n_hidden) + "_" + str(lambda_param) + ".png"
    name_filter ="autoencoderfilter_" + str(n_hidden) + "_" + str(lambda_param) + ".png"

    problem_24(auto_encoder.get_reconstructed_input(
        auto_encoder.get_hidden_values(train_set[:100])).eval(), 10, 10, 28, 28, name=name_rec)
    problem_25(auto_encoder.weights.get_value(), int(numpy.sqrt(n_hidden)), int(numpy.sqrt(n_hidden)),
               28, 28, name=name_filter)

def train_climin(data, epochs=10, activation=tensor.nnet.sigmoid, name="errors.png",
                 optimizer="rmsprop", sparsity=True, lambda_param=0.1, nhidden=100, kl=False):
    n_hidden = 100

    train_set = data

    x = tensor.matrix('x')

    batch_size = 200

    n_train_batches = train_set.get_value().shape[0] // batch_size

    args = climin.util.iter_minibatches([train_set.get_value()], batch_size, [0, 0])
    args = ((i, {}) for i in args)

    index = tensor.lscalar()
    rng = numpy.random.RandomState(1337)
    n_hidden = nhidden
    auto_encoder = AutoEncoder(
        x=x,
        numpy_rng=rng,
        n_visible=train_set.get_value().shape[1],
        n_hidden=n_hidden,
        lambda_param=lambda_param,
        sparse_cost=sparsity,
        kl_cost=kl
    )

    if optimizer is "sgd":
        print "Gradient Descent"
        climin_optimizer = climin.GradientDescent(auto_encoder.get_params(), auto_encoder.loss_grad,
                                                  step_rate=0.1, momentum=.95, args=args)
    else:
        print "rmsprop"
        climin_optimizer = climin.RmsProp(auto_encoder.get_params(), auto_encoder.loss_grad,
                                          step_rate=0.007, decay=0.9, args=args)

    current_run = 0
    c = []

    for info in climin_optimizer:
        # go through trainng set
        if current_run > epochs * n_train_batches:
            break

        c.append(auto_encoder.loss_func(auto_encoder.get_params(),
                                        train_set.get_value()[(current_run % n_train_batches) * batch_size:
                                        ((current_run % n_train_batches) + 1) * batch_size]))

        current_run += 1

        if current_run % n_train_batches is 0 and current_run > 0:
            print 'Current Epoch: ', current_run / n_train_batches, '\t Current Cost: ', numpy.mean(c)
            # reset cost
            c = []

    name_rec = "autoencoderrec_rms_" + str(n_hidden) + "_" + str(lambda_param) + ".png"
    name_filter ="autoencoderfilter_rms_" + str(n_hidden) + "_" + str(lambda_param) + ".png"

    problem_24(auto_encoder.get_reconstructed_input(
        auto_encoder.get_hidden_values(train_set[:100])).eval(), 10, 10, 28, 28, name=name_rec)
    problem_25(auto_encoder.weights.get_value(), int(numpy.sqrt(n_hidden)), int(numpy.sqrt(n_hidden)),
               28, 28, name=name_filter)

def problem_24(values, rows, cols, height, width, name="autoencoderrec.png"):
    border = 1
    image = numpy.zeros((rows * height + border * rows, cols * width + border * cols))
    for number in range(values.shape[0]):
        start_row = height * (number / cols) + (number / cols + 1) * border
        start_col = width * (number % cols) + (number % cols + 1) * border
        image[start_row:start_row + height, start_col:start_col + width] = values[number, :].reshape(height, width)

    pyplot.figure(dpi=300)
    pyplot.imshow(image)
    pyplot.axis('off')
    pyplot.set_cmap('rainbow')
    pyplot.colorbar()
    pyplot.savefig(name, dpi=300)

    print values[:100]


def problem_25(values, rows, cols, height, width, name="autoencoderfilter.png"):
    border = 2
    image = numpy.zeros((rows * height + border * rows, cols * width + border * cols))

    for number in range(values.shape[1]):
        start_row = height * (number / cols) + (number / cols + 1) * border
        start_col = width * (number % cols) + (number % cols + 1) * border
        image[start_row:start_row + height, start_col:start_col + width] = values[:, number].reshape(height, width)

    pyplot.figure(dpi=300)
    pyplot.imshow(image)
    pyplot.set_cmap('gray')
    pyplot.axis('off')
    pyplot.savefig(name, dpi=300)

if __name__ == '__main__':
    train_data, valid_data, test_data = utils.loadMinstDataSet(current_dir + "../mnist.pkl.gz")
    trainx, trainy = train_data

    for n in [1024, 784, 400, 225, 100]:
        for k in [0.05, 0.2, 1]:
            train_climin(trainx, nhidden=n, sparsity=True, lambda_param=k)

    train_climin(trainx, optimizer="rmsprop", sparsity=False, lambda_param=0)
    train(trainx, sparsity=False, lambda_param=0)

    # KL Divergence
    for k in [0.04, 0.02]:
        for n in [1024, 225]:
            train_climin(trainx, nhidden=n, sparsity=False, kl=True, lambda_param=k)