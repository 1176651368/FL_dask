from communication.tools import ConnectLocal
import dask.array as da
data = da.random.random_sample((3,3))
c = ConnectLocal(role='guest')
c.push(data,role='host')