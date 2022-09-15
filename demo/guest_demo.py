from FLmodel.LinearRegression.H.guest import LRGuest
import pandas as pd
import dask.array as da
data = pd.read_csv("../data/motor_hetero_guest.csv").values[:, 1:]
data = da.array(data)
# train
epoch = 10
Guest = LRGuest()
Guest.fit(x=data[:-50, 1:], y=data[:-50, 0:1], epoch=epoch)

# predict
Guest.predict(x=data[-50:, 1:])
