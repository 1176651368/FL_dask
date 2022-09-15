import numpy as np

from FLmodel.LinearRegression.H.host import LRHost
import pandas as pd
import dask.array as da
from encryptor.paillier.paillier import generate_paillier_keypair
data = pd.read_csv("../data/motor_hetero_host.csv").values[:, 1:]
data = da.array(data)
data = da.random.random_sample((3,3))
p,q = generate_paillier_keypair()
data = p.encrypt(data)
data = q.decrypt(data).compute()
# print(data)
# input()
# train
epoch = 10
Host = LRHost()
Host.fit(x=data[:-50, :], epoch=epoch)

# predict
Host.predict(x=data[-50:, :])
