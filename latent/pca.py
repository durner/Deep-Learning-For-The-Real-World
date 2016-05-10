import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
sys.path.append(current_dir  + '..')

import theano
import theano.tensor as tensor
import theano.tensor.nlinalg as linalg
import matplotlib.pyplot as plt
from itertools import product
import os
import utils


class PCA(object):
    def __init__(self, x, components=None):
        self.x = x

        # zero mean
        self.x -= tensor.mean(self.x, axis=0)

        # calculate covariance matrix
        n, dim = self.x.shape
        self.cov = tensor.dot(tensor.transpose(self.x), self.x) / (dim - 1)

        # compute eigenvectors
        v, w = linalg.eigh(self.cov)

        v, w = v[::-1], w[:, ::-1]


        if components != None:
            v = v[:components]
            w = w[:, :components]

        self.w = w
        self.v = v

        # compute pca of input
        self.output = tensor.dot(self.x, self.w)


def train(trainx, trainy, name):
    x = tensor.matrix('x')

    pca = PCA(x, components=2)


    train_model = theano.function(
        inputs=[x],
        outputs=pca.output,
    )

    X_train, y_train = trainx / 255., trainy

    print "pca running..."

    fig, plots = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    plt.prism()
    for i, j in product(xrange(10), repeat=2):
        if i > j:
            continue
        X_ = X_train[(y_train == i) + (y_train == j)]
        y_ = y_train[(y_train == i) + (y_train == j)]
        X_transformed = train_model(X_)
        print "pca " , i,  " and " , j
        plots[i, j].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
        plots[i, j].set_xticks(())
        plots[i, j].set_yticks(())

        plots[j, i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
        plots[j, i].set_xticks(())
        plots[j, i].set_yticks(())
        if i == 0:
            plots[i, j].set_title(j)
            plots[j, i].set_ylabel(j)

    plt.tight_layout()
    plt.savefig(name)

if __name__ == '__main__':
    train_data, valid_data, test_data = utils.loadMinstDataSet(current_dir + "../mnist.pkl.gz")
    trainx, trainy = valid_data
    train(trainx.eval(), trainy.eval(), "scatterplotMNIST.png")

    trainx, trainy = utils.load_cifar_file(os.path.join(current_dir + "../cifar-10-batches-py/", 'test_batch'))
    train(utils.convert_to_grayscale(trainx), trainy, "scatterplotCIFAR.png")