import pickle
import time
import numpy as np
import os

class ConnectLocal(object):

    def __init__(self, role):
        self.role = role

    def get(self, role='guest') -> np.ndarray:
        """
        :param role: get from who
        :return:
        """
        file_name = f"{role}-{self.role}.pkl"
        while not os.path.exists(file_name):
            time.sleep(0.5)
        data = pickle.load(open(file_name, 'rb'))
        os.remove(file_name)
        return data

    def push(self, data, role='host'):
        """
        :param data: send data
        :param role: send to who
        :return:
        """
        file_name = f"{self.role}-{role}.pkl"
        while os.path.exists(file_name):
            time.sleep(1)
        pickle.dump(obj=data, file=open(file_name, 'wb'))
