import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

N = 1000  # number of data
d = 10  # size of feature
beta = np.random.randint(5, 10, size=(d,))  # Targe value of Theta, It can be any value
print("beta: ", beta)
Data = []
test = []
while len(Data) < N:
    x = np.random.random(d)  # random number btw 0 and 1
    y = np.dot(beta, x) + 3 * (random.random() - 0.5)  # our target data
    Data.append((x, y))
    test.append(y)

# print("Data: ", Data)

# Cost function: J(theta) = 1/2N * sum((y - theta*x)^2)

theta = np.random.randint(5, 10, size=(d,))  # initial value of theta


def j(theta, Data):
    cost = 0
    for x, y in Data:
        cost += 0.5 * (y - np.dot(theta, x)) ** 2
    return cost / len(Data)


# With this cost function, fill in the code below such that parameter theta is trained to be its target value(beta)
# using gradient descent algorithm.
def gradJ(theta, Data):
    grad = np.zeros(len(theta))
    for x, y in Data:
        grad += (np.dot(theta, x) - y) * x
    return grad / len(Data)


alpha = 0.1  # learning rate
while True:  # gradient descent
    theta = theta - alpha * gradJ(theta, Data)
    if j(theta, Data) < j(beta, Data):
        break
print("Rounded theta: ", np.around(theta, decimals=2))
print("Actual theta: ", theta)
print("Difference between beta and theta: ", beta - theta)
#Plot theta
plt.plot(theta, label='theta')
plt.plot(beta, label='beta')
plt.legend()
plt.show()

