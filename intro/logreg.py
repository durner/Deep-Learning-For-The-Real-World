import numpy
import theano
import theano.tensor as tensor
rng = numpy.random

D = (rng.randn(400, 784), rng.randint(size=400, low=0, high=2))
training_steps = 1000

x = tensor.dmatrix("x")
y = tensor.dvector("y")

w = theano.shared(rng.randn(784), name="w")

b = theano.shared(0., name="b")

p_1 = 1 / (1 + tensor.exp(-tensor.dot(x, w) - b))
prediction = p_1 > 0.5
xent = -y * tensor.log(p_1) - (1-y) * tensor.log(1-p_1)
cost = xent.mean() + 0.01 * (w ** 2).sum()
gw, gb = tensor.grad(cost, [w, b]) 

train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

for i in range(training_steps):
    pred, err = train(D[0], D[1])

print(w.get_value())
print(b.get_value())
print(D[1])
print(predict(D[0]))
