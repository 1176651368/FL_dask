import numpy as np
from communication.tools import ConnectLocal
import random
import dask.array as da

class LRBase(object):

    def __init__(self, role: str = 'host', lr: float = 0.05):
        self.role = role
        self.lr = lr

        self.x = None
        self.y = None

        self.connect = ConnectLocal(role=role)
        self.weight = None
        self.bias = None

        self.noise = None
        self.public_key = None

        self.d = None

    def __str__(self):
        if self.weight is None:
            return "no model detail , should fit first"
            # setattr(self.weight, 'shape', None)
            # setattr(self.bias, 'shape', None)
        model_detail = f"model detail :\n\tlr: {self.lr}\n\t" \
                       + f'weight shape:{self.weight.shape}\n\t' \
                       + f'bias shape:{self.bias.shape}\n\t '

        return model_detail

    def _get_local_r(self) -> np.ndarray:
        # return self.x
        return da.dot(self.x, self.weight) + self.bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise

    def backward(self):
        # get grad_noise from arbiter
        grad_noise = self.connect.get('arbiter')
        # Gradient noise removal
        grad = grad_noise - self.noise

        grad = da.pad(grad, pad_width=((0, 0), (0, self.x.shape[1] - 1)), mode='edge')
        # update weight and bias
        self.weight = self.weight - (self.lr * 2 * (grad * self.x).mean(axis=0)).reshape((-1, 1))
        self.bias = self.bias - self.lr * grad.mean()

    def init_weight(self, shape):
        """
        :param shape: data shape
        :return:
        """
        self.weight = da.random.random_sample((shape[1], 1))
        self.bias = da.random.random_sample((1, 1))
        print(self)

    def forward_step_2(self):
        self.noise = random.random()
        en_grad_noise = self.d + self.noise
        # send grad with noise to arbiter
        self.connect.push(en_grad_noise, 'arbiter')

    def predict(self, x):
        setattr(self, 'x', x)
        self.connect.push(self._get_local_r(), 'arbiter')

    def compute(self):
        self.weight = self.weight.compute()
        self.bias = self.bias.compute()