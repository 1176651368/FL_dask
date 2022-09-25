import os.path
import pickle
import sys
import time
import numpy as np


class ConnectLocal(object):

    def __init__(self, role):
        self.role = role
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")

    def get(self, role='guest') -> np.ndarray:
        """
        :param role: get from who
        :return:
        """
        import datetime
        # st = datetime.datetime.now()
        file_name = f"./tmp/{role}-{self.role}.pkl"
        while not os.path.exists(file_name):
            time.sleep(0.1)
        file = open(file_name, 'rb')

        data = pickle.load(file)
        os.remove(file_name)
        # print("get cost",datetime.datetime.now() - st)
        return data

    def push(self, data, role='host'):
        """
        :param data: send data
        :param role: send to who
        :return:
        """
        import datetime
        # st = datetime.datetime.now()
        file_name = f"./tmp/{self.role}-{role}.pkl"
        while os.path.exists(file_name):
            time.sleep(0.5)
        pickle.dump(obj=data, file=open(file_name, 'wb'))
        # print("send cost",datetime.datetime.now() - st)
