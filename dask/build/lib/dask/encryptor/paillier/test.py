import numpy as np
import datetime


def np_paillier(data):
    from np_paillier.paillier import Paillier
    p, q = Paillier().generate_paillier_keypair(n_length=1024)
    st = datetime.datetime.now()
    en_data = p.encrypt(data)
    print(f"encrypt cost:{datetime.datetime.now() - st}")

    st = datetime.datetime.now()
    de_data = q.decrypt(en_data)
    print(f"decrypt cost:{datetime.datetime.now() - st}")
    # print("origin data:",data[0][0:3])
    # print("decrypt data:",de_data[0][0:3])
    print(data[0][0:3] == de_data[0][0:3])


def phe_paillier(data):
    from phe.paillier import generate_paillier_keypair
    p, q = generate_paillier_keypair(n_length=1024)
    st = datetime.datetime.now()
    en_data = [p.encrypt(i) for i in data.flatten().tolist()]
    en_data_array = np.array(en_data).reshape(data.shape)
    print(f"encrypt cost:{datetime.datetime.now() - st}")

    st = datetime.datetime.now()
    de_data = [q.decrypt(i) for i in en_data]
    de_data_array = np.array(de_data).reshape(data.shape)
    print(f"decrypt cost:{datetime.datetime.now() - st}")
    print(data[0][0:3] == de_data_array[0][0:3])


data = np.random.random_sample((100, 100))
np_paillier(data)
print("*" * 10)
phe_paillier(data)
