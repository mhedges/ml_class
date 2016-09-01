import matplotlib
import matplotlib.pyplot as plt

from pylab import plot, ylim
from random import choice
from numpy import array, dot, random

def step(x):
    if x < 0 :
        return 0
    else:
        return 1

# 2D implementation
training_data = [
    (array([1,1]), 0),
    (array([2,2]), 0),
    (array([0,1]), 1),
    (array([1,2]), 1),
]
w = random.rand(2)

errs = []
theta = 0.25
n = 60

for i in range(n):
    x, predicted = choice(training_data)
    calculated = dot(w, x)
    err = predicted - step(calculated)
    errs.append(err)
    w += theta * err * x

for x, y in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, step(result)))

ylim([-1.5,1.5])
plot(errs)
