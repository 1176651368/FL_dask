from .base import LRBase
from encryptor.paillier.paillier import toArray

class LRHost(LRBase):

    def __init__(self):
        super(LRHost, self).__init__(role="host")

    def forward_step_1(self):
        # send [[y]] = wx+b and [[y**2]] to guest
        out = self._get_local_r()
        en_local_r = toArray(self.public_key.encrypt(out))
        en_local_r2 = toArray(self.public_key.encrypt(out * out))
        self.connect.push([en_local_r, en_local_r2], role='guest')

    def forward_step_2(self):
        # get d from guest
        self.d = self.connect.get('guest')
        super(LRHost, self).forward_step_2()

    def fit(self, x, epoch=10):
        self.init_weight(x.shape)
        self.public_key = self.connect.get('arbiter')
        setattr(self, 'x', x)
        for i in range(epoch):
            self.forward_step_1()
            self.forward_step_2()
            self.backward()
