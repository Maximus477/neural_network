# This ws just a test to better understand how a neural network works.

from numpy import exp, array, random, dot

# training set, in order
training_set_inputs = array([[1, 2011, 5, 3, 6800, 1, 0], [1, 2010, 4, 2, 4500, 1, 1], [2, 1982, 4, 2, 9392, 0, 1], [2, 1975, 4, 2, 5400, 0, 1]])

# results of the training set (in order)
training_set_outputs = array([[775000, 594100, 595000, 580000]]).T

# initialize random
random.seed(1)

# number of neurons, 7 = input, 1 = output
synaptic_weights = 2 * random.random((7, 1)) - 1

# output 
for iteration in range(5):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
result = 1 / (1 + exp(-(dot(array([2, 1982, 4, 2, 7370, 0, 2]), synaptic_weights))))
print(result)

# This code doesn't work for multiple reasons, but most importantly
# the inputs and outputs have to be normalized between 0 and 1, to
# have a comparable scale of each input. See main_ez.py for finished code