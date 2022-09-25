import numpy as np

from FLmodel.LinearRegression.H.guest import LRGuest
import dask.array as da
from dask.distributed import Client
import dask.dataframe as df
import pandas as pd
if __name__ == '__main__':
    Client()
    data = np.random.random_sample((100,101))
    # data = pd.read_csv("../../data/motor_hetero_guest.csv").values[:, 1:]
    epoch = 10
    Guest = LRGuest()
    x = da.array(data[:, 1:])
    y = da.array(data[:, 0:1])
    Guest.fit(x,y, epoch=epoch)

    # predict
    Guest.predict(x=data[:, 1:])
