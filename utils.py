import numpy
import gzip
import pickle
import os
import scipy
import scipy.ndimage as srot
import random
import theano
import theano.tensor as tensor

def load_MNIST(filename):
    with gzip.open(filename, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    return train_set, valid_set, test_set

def loadMinstDataSet(filename, augument=False):
    def shared_dataset(data_xy, augument=False, borrow=True):
        data_x, data_y = data_xy

        data_x_npy = numpy.asarray(data_x, dtype=theano.config.floatX)
        data_y_npy = numpy.asarray(data_y, dtype=theano.config.floatX)

        if augument:
            data_x_npy = numpy.concatenate((data_x_npy, data_x_npy))
            data_y_npy = numpy.concatenate((data_y_npy, data_y_npy))

            for x in range(data_x_npy.shape[0] / 2):
                sqrt = numpy.sqrt(data_x_npy.shape[1])
                rot = random.uniform(5, -5)
                rotated = numpy.reshape(data_x_npy[x, :], (sqrt, sqrt))
                rotated = srot.rotate(rotated, rot)
                rotated = rotated[(rotated.shape[0] - sqrt) / 2: (rotated.shape[0] - sqrt) / 2 + sqrt,
                          (rotated.shape[1] - sqrt) / 2: (rotated.shape[1] - sqrt) / 2 + sqrt]
                data_x_npy[x, :] = numpy.reshape(rotated, (sqrt * sqrt,))

        print data_x_npy.shape

        # add required 1 vector for bias
        # shared_x = theano.shared(numpy.hstack((data_x_npy,
        #                                       numpy.ones((data_x_npy.shape[0], 1), dtype=tensor.dscalar()))),
        #                          borrow=borrow)
        shared_x = theano.shared(data_x_npy, borrow=borrow)

        shared_y = theano.shared(data_y_npy, borrow=borrow)

        return shared_x, tensor.cast(shared_y, 'int32')

    train_set, valid_set, test_set = load_MNIST(filename)

    return shared_dataset(train_set, augument), shared_dataset(valid_set), shared_dataset(test_set)


def load_cifar_file(filename):
    with open(filename, 'rb') as f:
        dict= pickle.load(f)
        X = dict['data']
        Y = dict['labels']
        X = numpy.array(X).astype('float64')
        Y = numpy.array(Y).astype('int32')
        return X, Y

def convert_to_grayscale(val, shrinked=False):
    X = val.reshape(val.shape[0], 3, 32, 32).mean(1).reshape(val.shape[0], -1)
    if shrinked:
        x_ret = numpy.zeros((X.shape[0], 12 * 12), dtype='float64')
        for k in range(X.shape[0]):
            x_ret[k, :] = scipy.misc.imresize(X[k, :].reshape((32, 32)), size=(12, 12)).reshape((12*12))
    else:
        x_ret = X
    return x_ret


def load_cifar(directory, shrinked=False):
    xs = []
    ys = []
    for k in range(1,6):
        f = os.path.join(directory, "data_batch_%d" % k)
        X, Y = load_cifar_file(f)
        xs.append(convert_to_grayscale(X, shrinked))
        ys.append(Y)
    trainx = numpy.concatenate(xs)
    trainy = numpy.concatenate(ys)
    testx, testy = load_cifar_file(os.path.join(directory, 'test_batch'))
    return trainx, trainy, convert_to_grayscale(testx, shrinked), testy