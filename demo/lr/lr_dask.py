import dask.dataframe as df
import dask.array as da
from Data.H.HDataFrame import HDataFrame


class Lr:

    def __init__(self, learning_rate: float = 0.01, itr=10):
        self.lr = learning_rate
        self.w = None
        self.b = None
        self.itr = itr
        self.x = None
        self.y = None
        self.pred = None

    def forward(self):
        self.pred = da.dot(self.x, self.w) + self.b

    def backward(self):
        grad = self.grad_loss()
        self.w = self.w - (self.lr * grad * self.x).mean(axis=0)
        self.b = self.b - (self.lr * grad).mean()

    def loss(self):
        return (1 / 2 * (self.pred - self.y) ** 2).mean()

    def grad_loss(self):
        return self.pred - self.y

    def init(self, x, y):
        self.x = x
        self.y = y

        self.w = da.random.random_sample((x.shape[1], y.shape[1]))
        self.b = da.random.random_sample((y.shape[1], 1))

    def fit(self, x, y):
        self.init(x, y)
        for i in range(self.itr):
            self.forward()
            print(self.loss().compute())
            self.backward()

    def compute(self):
        self.w = self.w.compute()
        self.b = self.b.compute()


data = df.read_csv("../data/motor_hetero_guest.csv")
data = HDataFrame(data, role='guest')
lr = Lr()
lr.fit(data.feature, data.label)
lr.compute()
