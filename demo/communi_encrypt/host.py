from communication.tools import ConnectLocal

c = ConnectLocal(role='host')
data = c.get(role='guest')
print(data.compute())