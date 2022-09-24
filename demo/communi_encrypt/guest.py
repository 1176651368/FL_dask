from communication.tools import ConnectLocal
import dask.array as da
from encryptor.paillier.paillier import generate_paillier_keypair,toArray
p,q = generate_paillier_keypair()
data = da.random.random_sample((3,3))
data2 = toArray(p.encrypt(data))
c = ConnectLocal(role='guest')
c.push([data2,q],role='host')
print(data.compute())
