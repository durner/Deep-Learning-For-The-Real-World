import numpy
import theano.tensor as T
from theano import function
import theano

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)
print f(2, 3)

x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
print f([[1, 2], [3, 4]], [[10, 20], [30, 40]])


x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
print f([[1, 2], [3, 4]], [[100, 200], [300, 400]])


x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s)
print logistic([[0, 1], [-1, -2]])


a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = theano.function([a, b], [diff, abs_diff, diff_squared])
print f([[1, 1], [1, 1]], [[0, 1], [2, 3]])


from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])
accumulator(1)
print state.get_value()
accumulator(300)
print state.get_value()

decrementor = function([inc], state, updates=[(state, state-inc)])
decrementor(2)
print state.get_value()

copy_accumulator = accumulator.copy()
copy_accumulator(9000)
print state.get_value()



