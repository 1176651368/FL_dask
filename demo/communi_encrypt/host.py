from communication.tools import ConnectLocal

c = ConnectLocal(role='host')
[data,q] = c.get(role='guest')
print(q.decrypt(data).compute())
