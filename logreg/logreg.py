import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
sys.path.append(current_dir  + '..')

import theano
import theano.tensor as tensor
import numpy
import climin
import climin.util
import utils
from matplotlib import pyplot


# The following functions are in general independent of a specific classifier


def update_weights(cost_function, weights, learning_rate):
    update_gradient = theano.grad(cost=cost_function, wrt=weights)
    updates = [(weights, weights - learning_rate * update_gradient)]
    return updates

# Expects the input as dim + 1, where the extra dimension
# must be filled with 1s to be in the space n + bias -> n + 1
class LogisticRegression(object):
    # Must be in numpy format with train, test, and validation set
    def __init__(self, x, y, n_dim, k_classes):
        self.weights = theano.shared(value=numpy.zeros(
            (n_dim, k_classes),
            # (n_dim , k_classes),
            dtype=tensor.dscalar()
        ), name="weights")

        self.bias = theano.shared(
            value=numpy.zeros((k_classes,), dtype=theano.config.floatX),
            name='bias')

        self.n_dim = n_dim
        self.classes = k_classes

        self.x = x
        self.y = y

        self.probability_d_in_k = tensor.nnet.softmax(theano.dot(self.x, self.weights) + self.bias)
        self.classification = tensor.argmax(self.probability_d_in_k, axis=1)

        self.template = [(self.n_dim, self.classes), (self.classes,)]

        self.loss_gradient = theano.function(
            inputs=[self.x, self.y],
            outputs=[tensor.grad(self.log_loss(), self.weights), tensor.grad(self.log_loss(), self.bias)]
        )
        self.loss_overall = theano.function(
            inputs=[self.x, self.y],
            outputs=self.log_loss(),
        )

    def log_loss(self):
        return -tensor.mean(tensor.log(self.probability_d_in_k)[tensor.arange(self.y.shape[0]), self.y])

    def zero_one_loss(self):
        if self.y.ndim == self.classification.ndim and self.y.dtype.startswith('int'):
            # run over all classification values and check whether it matches the result
            # inherently cast the bool to integer to mean over the result
            # (Theano has no boolean dtype. Instead, all boolean tensors are represented in 'int8'.)
            # tensor.mean(tensor.cast(tensor.neq(classification, result_vector), dtype='int8'))
            return tensor.mean(tensor.neq(self.classification, self.y))
        else:
            raise TypeError()

    def set_params(self, params):
        [w, b] = climin.util.shaped_from_flat(params, self.template)
        self.weights.set_value(w)
        self.bias.set_value(b)

    def get_params(self):
        flat, (w, b) = climin.util.empty_with_views(self.template)
        return numpy.concatenate([self.weights.get_value().flatten(), self.bias.get_value().flatten()])

    def loss_grad(self, params, input, target):
        self.set_params(params)
        return numpy.concatenate([grad.flatten() for grad in self.loss_gradient(input, target)])

    def loss_func(self, params, input, target):
        self.set_params(params)
        return self.loss_overall(input, target)


def train(data, report_errors):
    train_set, valid_set, test_set = data
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set

    training_error = []
    validation_error = []
    test_error = []

    index = tensor.lscalar("index")
    x = tensor.matrix('x')
    y = tensor.ivector('y')

    learning_rate = 0.13
    batch_size = 600
    epochs = 300

    classifier = LogisticRegression(x=x, y=y, n_dim=28 * 28, k_classes=10)

    train_model = theano.function(
        inputs=[index],
        outputs=classifier.log_loss(),
        updates=update_weights(classifier.log_loss(), classifier.weights, learning_rate),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_error_model = theano.function(
        inputs=[index],
        outputs=classifier.zero_one_loss(),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validation_model = theano.function(
        inputs=[index],
        outputs=classifier.zero_one_loss(),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    n_train_batches = train_set_x.get_value().shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value().shape[0] // batch_size

    current_epochs = 0
    best_loss = numpy.inf
    best_weights = classifier.weights
    best_bias = classifier.bias
    stop_count = 0

    while current_epochs < epochs and not stop_count > 20:
        for batch_index in range(0, n_train_batches):
            train_model(batch_index)

        # check on validation
        current_validation_loss = 0
        for batch_index in range(0, n_valid_batches):
            current_validation_loss += validation_model(batch_index)
        current_validation_loss /= n_valid_batches
        validation_error.append(current_validation_loss)

        if report_errors:
            #check on test
            test_error.append(test(classifier, data))

            # check on train
            current_train_loss = 0
            for batch_index in range(0, n_train_batches):
                current_train_loss += train_error_model(batch_index)
            current_train_loss /= n_train_batches
            training_error.append(current_train_loss)

        if current_validation_loss < best_loss:
            best_loss = current_validation_loss
            best_bias = classifier.bias.get_value()
            best_weights = classifier.weights.get_value()
            stop_count = 0
        else:
            stop_count += 1

        print(
            'Current Epoch: %i \t Current Validation Error: %f %%' %
            (
                current_epochs,
                best_loss * 100.
            )
        )
        current_epochs += 1

    classifier.weights = best_weights
    classifier.bias = best_bias

    if report_errors:
        problem_12(training_error, validation_error, test_error)

    return classifier, best_loss


def train_climin(data, report_errors, algorithm='sgd'):
    train_set, valid_set, test_set = data
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set

    training_error = []
    validation_error = []
    test_error = []

    index = tensor.lscalar("index")
    x = tensor.matrix('x')
    y = tensor.ivector('y')

    batch_size = 600
    epochs = 300

    n_train_batches = train_set_x.get_value().shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value().shape[0] // batch_size

    args = climin.util.iter_minibatches([train_set_x.eval(), train_set_y.eval()], batch_size, [0, 0])
    args = ((i, {}) for i in args)

    classifier = LogisticRegression(x=x, y=y, n_dim=28 * 28, k_classes=10)

    if algorithm == "sgd":
        print "GradientDescent"
        name = "errors_sgd.png"
        climin_optimizer = climin.GradientDescent(classifier.get_params(), classifier.loss_grad, step_rate=0.1, args=args)
    else :
        print "AdaDelta"
        name = "errors_adadelta.png"
        climin_optimizer = climin.Adadelta(classifier.get_params(), classifier.loss_grad, args=args)

    validation_model = theano.function(
        inputs=[index],
        outputs=classifier.zero_one_loss(),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    train_error_model = theano.function(
        inputs=[index],
        outputs=classifier.zero_one_loss(),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    current_run = 0
    best_loss = numpy.inf
    best_weights = classifier.weights
    best_bias = classifier.bias
    stop_count = 0

    for info in climin_optimizer:
        if current_run > epochs * n_train_batches or stop_count > 20:
            break

        current_run += 1

        # check on validation
        if current_run % n_train_batches is 0 and current_run > 0:
            current_validation_loss = 0
            for batch_index in range(0, n_valid_batches):
                current_validation_loss += validation_model(batch_index)
            current_validation_loss /= n_valid_batches
            validation_error.append(current_validation_loss)

            if report_errors:
                # check on test
                test_error.append(test(classifier, data))

                # check on train
                current_train_loss = 0
                for batch_index in range(0, n_train_batches):
                    current_train_loss += train_error_model(batch_index)
                current_train_loss /= n_train_batches
                training_error.append(current_train_loss)


            if current_validation_loss < best_loss:
                best_loss = current_validation_loss
                best_bias = classifier.bias.get_value()
                best_weights = classifier.weights.get_value()
                stop_count = 0
            else:
                stop_count += 1

            print(
                'Current Epoch: %i \t Current Validation Error: %f %%' %
                (
                    current_run / n_train_batches,
                    best_loss * 100.
                )
            )

    classifier.weights = best_weights
    classifier.bias = best_bias

    if report_errors:
        problem_12(training_error, validation_error, test_error, name=name)

    return classifier, best_loss

def test(classifier, data):
    index = tensor.lscalar("index")
    x = classifier.x
    y = classifier.y

    batch_size = 600

    train_set, valid_set, test_set = data
    test_set_x, test_set_y = test_set
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.zero_one_loss(),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    n_test_batches = test_set_x.get_value().shape[0] // batch_size

    loss = 0
    for batch_index in range(0, n_test_batches):
        loss += test_model(batch_index)
    loss /= n_test_batches
    return loss


def problem_8():
    data = utils.loadMinstDataSet(current_dir + "../mnist.pkl.gz", augument=True)
    classifier, smallest_validation_loss = train(data, True)
    print(
        'Test Set Error: %f %%' %
        (
            test(classifier, data) * 100.
        )
    )
    problem_11(classifier.weights)


def problem_10(algorithm='sgd'):
    data = utils.loadMinstDataSet(current_dir + "../mnist.pkl.gz", augument=True)
    classifier, smallest_validation_loss = train_climin(data, True, algorithm=algorithm)
    print(
        'Test Set Error: %f %%' %
        (
            test(classifier, data) * 100.
        )
    )
    name = "repflds_" + algorithm + ".png"
    problem_11(classifier.weights, name=name)


def problem_11(weights, rows=5, cols=2, height=28, width=28, name="repflds.png"):
    border = 2

    image = numpy.zeros((rows * height + border * rows, cols * width + border * cols))

    for number in range(weights.shape[1]):
        start_row = height * (number / cols) + (number / cols + 1) * border
        start_col = width * (number % cols) + (number % cols + 1) * border
        image[start_row:start_row + height, start_col:start_col + width] = weights[:, number].reshape(height, width)

    pyplot.figure(dpi=300)
    pyplot.imshow(image)
    pyplot.axis('off')
    pyplot.set_cmap('rainbow')
    pyplot.colorbar()
    pyplot.tight_layout()
    pyplot.savefig(name, dpi=300)


def problem_12(training_error, validation_error, test_error, name="errors.png"):
    pyplot.figure()
    pyplot.plot(range(1, len(training_error)+1), training_error, label="Train Error")
    pyplot.plot(range(1, len(validation_error)+1), validation_error, label="Validation Error")
    pyplot.plot(range(1, len(test_error)+1), test_error, label="Test Error")
    pyplot.xlabel("epochs")
    pyplot.ylabel("error")
    pyplot.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                  ncol=3, fancybox=True, shadow=True)
    pyplot.tight_layout()
    pyplot.savefig(name)

if __name__ == '__main__':
    problem_8()
    problem_10("sgd")
    problem_10("adadelta")
