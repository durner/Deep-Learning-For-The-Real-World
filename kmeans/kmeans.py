import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
sys.path.append(current_dir  + '..')

import theano
import theano.tensor as tensor
import numpy
from matplotlib import pyplot
import utils


def normalize_dict(D):
    lengths = numpy.sqrt(numpy.sum(D ** 2, axis=0))
    D /= lengths
    return D

class KMeansCoates(object):
    def __init__(self, x, dim, centers, batch_size, numpy_rng, eps_var=10, eps_whiten=0.01):
        self.x = x
        self.dim = dim
        self.centers = centers
        self.batch_size = batch_size
        self.rng = numpy_rng

        # uniformly start to initialize dict
        initial_dict = numpy.asarray(
            numpy_rng.uniform(
                low=0,
                high=1,
                size=(self.dim, self.centers)
            ),
            dtype=theano.config.floatX
        )

        # initial_dict = normalize_dict(initial_dict)
        self.D = theano.shared(value=normalize_dict(initial_dict), name='dictionary')


        self.x_normalized = x - tensor.mean(x, axis=0)
        self.x_normalized /= tensor.sqrt(tensor.var(self.x, axis=0) + eps_var)

        # calculate covariance matrix
        self.cov = tensor.dot(self.x_normalized.T, self.x_normalized) / float(dim - 1)

        self.W, self.V, _ = tensor.nlinalg.svd(self.cov) # maybe transpose the cov!

        # calculate the whitening
        self.step_between = tensor.dot(tensor.dot(self.W,
                                                  tensor.diag(float(1) / tensor.sqrt(self.V + eps_whiten))), self.W.T)


        self.x_whiten = tensor.dot(self.x_normalized, self.step_between)

        self.S = tensor.dot(self.x_whiten, self.D)

        self.update = tensor.dot(self.x_whiten.T, self.S)

    def return_result(self):
        return self.S, self.x_whiten


def problem_31(values, rows, cols, height, width, name="repflds.png"):
    border = 2
    print values.shape
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


def train(trainx, trainy, name, n_center):
    X_train, y_train = trainx / 255., trainy

    x = tensor.matrix('x')
    rng = numpy.random.RandomState(1337)

    kmeans = KMeansCoates(x, X_train.shape[1], n_center, X_train.shape[0], rng)


    train_model = theano.function(
        inputs=[x],
        outputs=kmeans.return_result(),
    )

    s = tensor.matrix('s')
    calculate_update = theano.function(
        inputs=[x, s],
        outputs=tensor.dot(x.T, s),
    )

    for k in range(10):
        print "run ", k
        S, x_whiten = train_model(X_train)

        # set argmax value only
        sprime = numpy.zeros((X_train.shape[0], n_center))
        argmax_k = numpy.argmax(numpy.abs(S), axis=1)
        for i in xrange(X_train.shape[0]):
            sprime[i, argmax_k[i]] =  1
        S *= sprime

        kmeans.D.set_value(normalize_dict(kmeans.D.get_value() + calculate_update(x_whiten, S)))


        # reinitialize empty clusters
        step = numpy.sum(S.T, axis=0)
        for i in range(n_center):
            if step[i] == 0:
                idx = numpy.random.randint(X_train.shape[1])
                kmeans.D[:, i].set_value(x_whiten[idx])

    problem_31(kmeans.D.get_value(), int(numpy.sqrt(n_center)), int(numpy.sqrt(n_center)), 12, 12)



if __name__ == '__main__':
    trainx, trainy, testx, testy = utils.load_cifar(current_dir + "../cifar-10-batches-py/", True)
    trainx = numpy.asarray(numpy.vstack((trainx, testx)), dtype=numpy.float64)
    trainy = numpy.hstack((trainy, testy))
    print "loaded!"
    train(trainx, trainy, "cifar.png", 400)
