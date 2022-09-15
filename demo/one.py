import dask.array as np
# import numpy as np

def forward(x, w, b):
    y = np.dot(x, w) + b
    return y


def backward(w, b, grad, data):
    w = w - (lr * grad * data).mean(axis=0)
    b = b - (lr * grad).mean()
    return w, b


def loss(y, t):
    return 1 / 2 * (y - t) ** 2


def grad_loss(y, t):
    return y - t


data = np.random.random_sample((10000, 3))
y = np.random.random_sample((10000, 1))

w = np.random.random_sample((3, 1))
b = np.random.random_sample((1, 1))

lr = 0.1
epoch = 100
import datetime
st = datetime.datetime.now()
for i in range(epoch):
    print(i)
    pred = forward(data, w, b)
    l = loss(pred, y).mean()
    grad = grad_loss(pred, y)
    w, b = backward(w, b, grad, data)
w.compute()
b.compute()
print(datetime.datetime.now() - st)

