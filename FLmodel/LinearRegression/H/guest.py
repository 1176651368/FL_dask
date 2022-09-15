from .base import LRBase


class LRGuest(LRBase):

    def __init__(self):
        super(LRGuest, self).__init__(role="guest")
        self.db = None

    def _compute_loss(self, y_host, y_host_2, y_guest):
        # La = y_host**2
        La = y_host_2.sum() / self.x.shape[0]
        # Lb = (y_guest-y)**2
        Lb = ((y_guest - self.y) ** 2).mean()
        # lab = y_host * (y_guest - y)
        Lab = 2 * (y_host * (y_guest - self.y)).sum() / self.x.shape[0]
        L = La + Lb + Lab

        self.connect.push([L, La, Lb, Lab], 'arbiter')

    def forward_step_1(self):
        [y_host, y_host_2] = self.connect.get(role='host')

        # guest local
        y_guest = self._get_local_r()

        # ya + yb - y
        self.d = y_host + y_guest - self.y
        self.connect.push(self.d, 'host')

        self._compute_loss(y_host, y_host_2, y_guest)

    def fit(self, x, y, epoch=10):
        assert x.shape[0] == y.shape[0], 'x.shape != y.shape, check your data!'

        self.init_weight(x.shape)
        self.public_key = self.connect.get('arbiter')
        setattr(self, 'x', x)
        setattr(self, 'y', y)
        for i in range(epoch):
            self.forward_step_1()
            self.forward_step_2()
            self.backward()



