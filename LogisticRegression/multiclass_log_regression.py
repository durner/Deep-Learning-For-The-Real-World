import theano
import theano.tensor as tensor
import numpy
import gzip
import pickle
import climin


# The following functions are in general independent of a specific classifier


def update_weights(cost_function, weights, learning_rate):
    update_gradient = theano.grad(cost=cost_function, wrt=weights)
    updates = [(weights, weights - learning_rate * update_gradient)]
    return updates


def log_loss(probability_d_in_k, result_vector):
    return -tensor.mean(tensor.log(probability_d_in_k)[tensor.arange(result_vector.shape[0]), result_vector])


def zero_one_loss(classification, result_vector):
    if result_vector.ndim == classification.ndim and result_vector.dtype.startswith('int'):
        # run over all classification values and check whether it matches the result
        # inherently cast the bool to integer to mean over the result
        # (Theano has no boolean dtype. Instead, all boolean tensors are represented in 'int8'.)
        # tensor.mean(tensor.cast(tensor.neq(classification, result_vector), dtype='int8'))
        return tensor.mean(tensor.neq(classification, result_vector))
    else:
        raise TypeError()


# Expects the input as dim + 1, where the extra dimension
# must be filled with 1s to be in the space n + bias -> n + 1
class LogisticRegression(object):
    # Must be in numpy format with train, test, and validation set
    def __init__(self, filename, n_dim, k_classes, epochs, batch_size, learn_loss, eval_loss, learning_rate=0.13):
        self.weights = theano.shared(value=numpy.zeros(
            (n_dim + 1, k_classes),
            # (n_dim , k_classes),
            dtype=tensor.dscalar()
        ), name="weights")

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learn_loss = learn_loss
        self.eval_loss = eval_loss

        def shared_dataset(data_xy, borrow=True):
            data_x, data_y = data_xy

            # add required 1 vector for bias
            shared_x = theano.shared(numpy.hstack((numpy.asarray(data_x, dtype=theano.config.floatX),
                                                   numpy.ones((data_x.shape[0], 1), dtype=tensor.dscalar()))),
                                     borrow=borrow)

            shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

            return shared_x, tensor.cast(shared_y, 'int32')

        with gzip.open(filename, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)

            self.train_set_x, self.train_set_y = shared_dataset(train_set)
            self.valid_set_x, self.valid_set_y = shared_dataset(valid_set)
            self.test_set_x, self.test_set_y = shared_dataset(test_set)

    def train(self):
        x = tensor.dmatrix('x')  # used as the current minibatch matrix
        y = tensor.ivector('y')  # used as the current minibatch label vector

        probability_d_in_k = tensor.nnet.softmax(theano.dot(x, self.weights))
        classification = tensor.argmax(probability_d_in_k, axis=1)

        log_loss_func = self.learn_loss(probability_d_in_k, y)
        zero_one_loss_func = self.eval_loss(classification, y)

        index = tensor.lscalar()
        train_model = theano.function(
            inputs=[index],
            outputs=log_loss_func,
            updates=update_weights(log_loss_func, self.weights, self.learning_rate),
            givens={
                x: self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: self.train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        validation_model = theano.function(
            inputs=[index],
            outputs=zero_one_loss_func,
            givens={
                x: self.valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: self.valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        n_train_batches = self.train_set_x.get_value().shape[0] // self.batch_size
        n_valid_batches = self.valid_set_x.get_value().shape[0] // self.batch_size

        current_epochs = 0
        best_loss = numpy.inf
        best_weights = self.weights
        stop_loop = False

        while current_epochs < self.epochs and not stop_loop:
            for batch_index in range(0, n_train_batches):
                train_model(batch_index)

            # check on validation
            current_validation_loss = 0
            for batch_index in range(0, n_valid_batches):
                current_validation_loss += validation_model(batch_index)
            current_validation_loss /= n_valid_batches

            if current_validation_loss < best_loss:
                best_loss = current_validation_loss
                best_weights = self.weights.get_value()
            print(
                'Current Epoch: %i \t Current Validation Error: %f %%' %
                (
                    current_epochs,
                    best_loss * 100.
                )
            )
            current_epochs += 1

        self.weights = best_weights

        return best_loss

    def test(self):
        x = tensor.dmatrix('x')  # used as the current minibatch matrix
        y = tensor.ivector('y')  # used as the current minibatch label vector
        index = tensor.lscalar()

        probability_d_in_k = tensor.nnet.softmax(theano.dot(x, self.weights))
        classification = tensor.argmax(probability_d_in_k, axis=1)

        zero_one_loss_func = self.eval_loss(classification, y)

        test_model = theano.function(
            inputs=[index],
            outputs=zero_one_loss_func,
            givens={
                x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: self.test_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        n_test_batches = self.test_set_x.get_value().shape[0] // self.batch_size

        loss = 0
        for batch_index in range(0, n_test_batches):
            loss += test_model(batch_index)
        loss /= n_test_batches
        return loss

if __name__ == '__main__':
    # loss_func = climin.
    classifier = LogisticRegression("/home/durner/Downloads/mnist.pkl.gz", n_dim=28 * 28, k_classes=10, epochs=5,
                                    batch_size=600, learn_loss=log_loss, eval_loss=zero_one_loss)

    smallest_validation_loss = classifier.train()
    print("Test Set Error: %f %%" % (classifier.test() * 100.))
