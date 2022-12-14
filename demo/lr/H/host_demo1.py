import datetime
import dask
import pandas as pd
import dask.array as da
from encryptor.paillier.paillier import generate_paillier_keypair, toArray
import numpy as np
from dask.distributed import Client

if __name__ == '__main__':
    client = Client()
    p, q = generate_paillier_keypair()
    # data = np.random.random_sample((1000,1000))
    # data = da.array(data)
    data = da.random.random_sample((10, 10), chunks=(10, 10))
    st = datetime.datetime.now()
    data2 = toArray(p.encrypt(data))
    data2 = q.decrypt(data2)
    s = data2.compute()
    print(s)
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
