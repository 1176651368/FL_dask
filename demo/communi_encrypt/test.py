
import dask.array as da
from communication.tools import ConnectLocal
data = da.random.random_sample((100,100))
conect = ConnectLocal(role="guest")
for i in range(0,10):
    #conect.push(data,role='host')
    data = data.dot(data)
   # print(type(data.__dask_graph__()))
    #print(dir(data.__dask_graph__()))
    # print(dir(data))
    # print(data.__dask_graph__())
    # print(data.__dask_keys__())
    #print(data.__dask_layers__())
    #print(data.__dask_graph__().layers)
    print(list(data.__dask_graph__().layers.keys()))
    s = data.__dask_graph__().cull(list(data.__dask_graph__().layers.keys())[0])
    print(dir(s))
    print(s.keys())
    print(s.values())
    #for i in .layers:
        #print(i)
    # print(data.__dask_postcompute__())
    # print(data.__dask_postpersist__())
    input()
    # for k,v in data.__dict__.items():
    #     print(k,"****",v)
    #     input()