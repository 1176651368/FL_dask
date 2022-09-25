import pandas as pd
import dask.array as da
from FLmodel.LinearRegression.H.host import LRHost
from dask.distributed import Client
import dask.dataframe as df
import numpy as np
if __name__ == '__main__':
    Client()
    #data = pd.read_csv("../../data/motor_hetero_guest.csv").values[:, 1:]
    data = np.random.random_sample((100, 100))
    epoch = 10
    Host = LRHost()
    Host.fit(x=da.array(data), epoch=epoch)

    # predict
    Host.predict(x=data)

