import datetime
import numpy as np
import mars.tensor as mt
import dask.array as da
st = datetime.datetime.now()
data = da.random.random_sample((10000000,100))
data = data.dot(da.random.random_sample((100,100)))
# data.execute()
r = data[0].compute()
print(r)
print(datetime.datetime.now() - st)
# 0:00:00.158977
# 0:00:00.306881