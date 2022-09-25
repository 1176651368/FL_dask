from .base import LRBase
import numpy as np
from encryptor.paillier.paillier import generate_paillier_keypair

class Arbiter(LRBase):

    def __init__(self):
        super(Arbiter, self).__init__(role='arbiter')
        self.loss: list = []

    def _init_key(self):
        """
            arbiter init public_key and private key
        """

        self.public_key, self.private_key = generate_paillier_keypair()
        self.connect.push(self.public_key, 'guest')
        self.connect.push(self.public_key, 'host')

    def fit(self, epoch=10):
        # init key and push key to guest and host
        self._init_key()
        print("push public key is success")
        for i in range(epoch):
            # get loss from guest and decrypt
            L = self.connect.get('guest')
            loss = self.private_key.decrypt(L)
            print("loss:", loss)

            self.loss.append(loss)
            # calculate grad , send grad to host and guest
            self.push_grad_to_guest_host()

    def push_grad_to_guest_host(self):
        """
        decrypt guest and host's grad ,and send to them
        :return:None
        """
        guest_grad_noise = self.connect.get('guest')
        guest_grad_noise_de = self.private_key.decrypt(guest_grad_noise)

        self.connect.push(guest_grad_noise_de, 'guest')

        host_grad_noise = self.connect.get('host')
        host_grad_noise_de = self.private_key.decrypt(host_grad_noise)
        self.connect.push(host_grad_noise_de, 'host')

    def predict(self, x=None) -> np.ndarray:
        pred_guest = self.connect.get('guest')
        pred_host = self.connect.get('host')
        y = pred_guest + pred_host
        return y

