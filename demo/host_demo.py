import datetime
from FL_dask.dask import dask
import pandas as pd
import FL_dask.dask.dask.array as da
from FL_dask.encryptor.paillier.paillier import generate_paillier_keypair, toArray
import numpy as np
from FL_dask.dask.dask.distributed import Client

if __name__ == '__main__':
    client = Client(n_workers=8, threads_per_worker=16)
    p, q = generate_paillier_keypair()
    # data = np.random.random_sample((1000,1000))
    # data = da.array(data)
    data = da.random.random_sample((100000, 100), chunks=(500, 100))
    st = datetime.datetime.now()
    data2 = p.encrypt(data)
    data2.compute()
    print(datetime.datetime.now() - st)
# st = datetime.datetime.now()
# data2 = data ** 2
# print(datetime.datetime.now() - st)


# if __name__ == '__main__':
#     dask.config.set(scheduler='multiprocessing',num_works=8)
#     data = pd.read_csv("../data/motor_hetero_host.csv").values[:, 1:]
#     data = da.from_array(data,chunks=(data.shape[0]//8,data.shape[1]))
#     p,q = generate_paillier_keypair()
#     st = datetime.datetime.now()
#     data = toArray(p.encrypt(data)).compute()
#     # print(data)
#     print(datetime.datetime.now() - st)

# data = da.random.random_sample(chunks=)
# st = datetime.datetime.now()
# data = q.decrypt(data).compute()
# print(datetime.datetime.now() - st)
# train
# epoch = 10
# Host = LRHost()
# Host.fit(x=data[:-50, :], epoch=epoch)
#
# # predict
# Host.predict(x=data[-50:, :])
