import pandas as pd
import dask.array as da
from FLmodel.LinearRegression.H.host import LRHost

data = pd.read_csv("../../data/motor_hetero_guest.csv").values[:, 1:]
data = da.array(data)
# from encryptor.paillier.paillier import generate_paillier_keypair,toArray
# p,q = generate_paillier_keypair()
# w = da.random.random_sample((data.shape[1],1))
# b = da.random.random_sample((1,1))
# data = data.dot(w)+b
# # print(data.compute()[0:5])
# # data = p.encrypt(data)
# # en_data = q.decrypt(toArray(data)).compute()
# # print(en_data[0:5])
# data = p.encrypt(b)
# en_data = q.decrypt(toArray(data)).compute()
# print(en_data)
epoch = 10
Host = LRHost()
Host.fit(x=data[:-50, :], epoch=epoch)

# predict
Host.predict(x=data[-50:, :])

