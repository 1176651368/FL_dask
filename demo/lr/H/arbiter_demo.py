import pandas as pd

from FLmodel.LinearRegression.H.arbiter import Arbiter
from matplotlib import pyplot as plt
from dask.distributed import Client
if __name__ == '__main__':
    Client()
    epoch = 3
    m = Arbiter()
    m.fit(epoch=epoch)

    # plt
    plt.figure()
    plt.plot(range(len(m.loss)), m.loss, color='red')
    plt.show()

    # predict
    pred = m.predict()

# # plt
# data = pd.read_csv("../../data/motor_hetero_guest.csv").values[:, 1:]
# y = data[-50:,0:1]
# plt.figure()
# plt.plot(range(len(pred)), pred, color='red')
# plt.bar(range(len(y)), y.flatten().tolist(), color='black')
# plt.show()