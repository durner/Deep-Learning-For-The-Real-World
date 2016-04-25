import numpy
import theano
import theano.tensor as tensor
import logreg.logreg as logreg
import climin
import climin.util
from matplotlib import pyplot


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, w=None, b=None, activation=tensor.tanh):
        self.input = input
        if w is None:
            w = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

        if b is None:
            b = numpy.zeros(
                (n_out,),
                dtype=tensor.dscalar()
            )

            if activation == tensor.nnet.sigmoid:
                w *= 4

            self.weights = theano.shared(value=w, name='weights_hidden', borrow=True)
            self.bias = theano.shared(value=b, name='bias_hidden', borrow=True)
        else:
            self.weights = w
            self.bias = b

        self.output = activation(tensor.dot(self.input, self.weights) + self.bias)
        self.template = [(n_in, n_out), (n_out,)]


# start-snippet-2
class NeronalNetwork(object):

    def __init__(self, rng, x, y, n_in, n_hidden, n_out):
        self.x = x
        self.y = y

        self.hidden_layer = HiddenLayer(
            rng=rng,
            input=self.x,
            n_in=n_in,
            n_out=n_hidden,
            activation=tensor.tanh
        )

        # The logistic regression layer gets as input the hidden units
        self.logreg_layer = logreg.LogisticRegression(
            x=self.hidden_layer.output,
            y = self.y,
            n_dim=n_hidden,
            k_classes=n_out
        )

        self.L1 = (
            abs(self.hidden_layer.weights).sum()
            + abs(self.logreg_layer.weights).sum()
        )

        self.L2_sqr = (
            (self.hidden_layer.weights ** 2).sum()
            + (self.logreg_layer.weights ** 2).sum()
        )

        self.params = [self.hidden_layer.weights, self.hidden_layer.bias, self.logreg_layer.weights,
                       self.logreg_layer.bias]

        self.gparams = [tensor.grad(self.logreg_layer.log_loss(), param) for param in self.params]

        self.template = self.hidden_layer.template + self.logreg_layer.template

        self.loss_gradient = theano.function(
            inputs=[self.x, self.y],
            outputs=self.gparams
        )
        self.loss_overall = theano.function(
            inputs=[self.x, self.y],
            outputs=self.logreg_layer.log_loss()
        )

    def set_params(self, params):
        [hw, hb, w, b] = climin.util.shaped_from_flat(params, self.template)
        self.logreg_layer.weights.set_value(w)
        self.logreg_layer.bias.set_value(b)
        self.hidden_layer.weights.set_value(hw)
        self.hidden_layer.bias.set_value(hb)

    def get_params(self):
        flat, (hw, hb, w, b) = climin.util.empty_with_views(self.template)
        return numpy.concatenate([self.hidden_layer.weights.get_value().flatten(),
                                  self.hidden_layer.bias.get_value().flatten(),
                                  self.logreg_layer.weights.get_value().flatten(),
                                  self.logreg_layer.bias.get_value().flatten()])

    def loss_grad(self, params, input, target):
        self.set_params(params)
        return numpy.concatenate([grad.flatten() for grad in self.loss_gradient(input, target)])

    def loss_func(self, params, input, target):
        self.set_params(params)
        return numpy.concatenate([overall.flatten() for overall in self.loss_overall(input, target)])


def train_l1_l2(data, report_errors):
    # configuration

    learning_rate = 0.01
    L1_reg = 0.00
    L2_reg = 0.0001
    epochs = 1
    batch_size = 20
    n_hidden = 300


    train_set, valid_set, test_set = data
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set

    training_error = []
    validation_error = []
    test_error = []

    index = tensor.lscalar("index")
    x = tensor.matrix('x')
    y = tensor.ivector('y')

    rng = numpy.random.RandomState(1337)

    classifier = NeronalNetwork(
        rng=rng,
        x=x,
        y=y,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    cost = (
        classifier.logreg_layer.log_loss()
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, classifier.gparams)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_error_model = theano.function(
        inputs=[index],
        outputs=classifier.logreg_layer.zero_one_loss(),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validation_model = theano.function(
        inputs=[index],
        outputs=classifier.logreg_layer.zero_one_loss(),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    n_train_batches = train_set_x.get_value().shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value().shape[0] // batch_size

    current_epochs = 0
    best_loss = numpy.inf
    best_settings = classifier.get_params()
    stop_loop = False

    while current_epochs < epochs and not stop_loop:
        for batch_index in range(0, n_train_batches):
            train_model(batch_index)

        # check on validation
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
            best_settings = classifier.get_params()
        print(
            'Current Epoch: %i \t Current Validation Error: %f %%' %
            (
                current_epochs,
                best_loss * 100.
            )
        )
        current_epochs += 1

    classifier.set_params(best_settings)

    if report_errors:
        problem_12(training_error, validation_error, test_error)

    return classifier, best_loss

def train_climin(data, report_errors):
    # configuration

    learning_rate = 0.01
    epochs = 10
    batch_size = 20
    n_hidden = 300

    train_set, valid_set, test_set = data
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set

    training_error = []
    validation_error = []
    test_error = []

    index = tensor.lscalar("index")
    x = tensor.matrix('x')
    y = tensor.ivector('y')

    n_train_batches = train_set_x.get_value().shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value().shape[0] // batch_size

    args = climin.util.iter_minibatches([train_set_x.eval(), train_set_y.eval()], batch_size, [0, 0])
    args = ((i, {}) for i in args)

    rng = numpy.random.RandomState(1337)

    classifier = NeronalNetwork(
        rng=rng,
        x=x,
        y=y,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    # climin_optimizer = climin.GradientDescent(classifier.get_params(), classifier.loss_grad, step_rate=0.02, momentum=.95, args=args)
    climin_optimizer = climin.RmsProp(classifier.get_params(), classifier.loss_grad, step_rate=0.02, args=args)

    validation_model = theano.function(
        inputs=[index],
        outputs=classifier.logreg_layer.zero_one_loss(),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    train_error_model = theano.function(
        inputs=[index],
        outputs=classifier.logreg_layer.zero_one_loss(),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    current_run = 0
    best_loss = numpy.inf
    best_settings = classifier.get_params()
    stop_loop = False

    for info in climin_optimizer:
        if current_run > epochs * n_train_batches or stop_loop:
            break
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
                best_settings = classifier.get_params()
            print(
                'Current Epoch: %i \t Current Validation Error: %f %%' %
                (
                    current_run / n_train_batches,
                    best_loss * 100.
                )
            )
        current_run += 1

    classifier.set_params(best_settings)

    if report_errors:
        problem_12(training_error, validation_error, test_error)

    return classifier, best_loss

def test(classifier, data):
    index = tensor.lscalar("index")
    x = classifier.x
    y = classifier.y

    batch_size = 20

    train_set, valid_set, test_set = data
    test_set_x, test_set_y = test_set
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.logreg_layer.zero_one_loss(),
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
    data = logreg.loadMinstDataSet("/home/durner/Downloads/mnist.pkl.gz", augument=False)
    classifier, smallest_validation_loss = train_l1_l2(data, True)
    print(
        'Test Set Error: %f %%' %
        (
            test(classifier, data) * 100.
        )
    )
    problem_11(classifier.hidden_layer.weights.eval(), 15, 20, 28, 28)

def problem_10():
    data = logreg.loadMinstDataSet("/home/durner/Downloads/mnist.pkl.gz", augument=False)
    classifier, smallest_validation_loss = train_climin(data, True)
    print(
        'Test Set Error: %f %%' %
        (
            test(classifier, data) * 100.
        )
    )
    problem_11(classifier.hidden_layer.weights.eval(), 15, 20, 28, 28)

def problem_11(weights, rows, cols, height, width):
    border = 2

    image = numpy.zeros((rows * height + border * rows, cols * width + border * cols))

    for number in range(weights.shape[1]):
        start_row = height * (number / cols) + (number / cols + 1) * border
        start_col = width * (number % cols) + (number % cols + 1) * border
        image[start_row:start_row + height, start_col:start_col + width] = weights[:, number].reshape(height, width)

    pyplot.figure(dpi=300)
    pyplot.imshow(image)
    pyplot.axis('off')
    pyplot.set_cmap('gist_ncar')
    pyplot.colorbar()
    pyplot.savefig('repflds.png', dpi=300)


def problem_12(training_error, validation_error, test_error):
    pyplot.figure()
    pyplot.plot(range(len(training_error)), training_error, label="Train Error")
    pyplot.plot(range(len(validation_error)), validation_error, label="Validation Error")
    pyplot.plot(range(len(test_error)), test_error, label="Test Error")
    pyplot.xlabel("epochs")
    pyplot.ylabel("error")
    pyplot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
                  fancybox=True, shadow=True, ncol=3)
    pyplot.savefig('errors.png')


if __name__ == '__main__':
    problem_10()