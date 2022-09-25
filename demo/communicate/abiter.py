from communication.tools import ConnectLocal

c = ConnectLocal(role='arbiter')
for i in range(5):
    c.push(data='a',role='guest')
    c.push(data='b',role='host')
    c.get(role='guest')
    c.get(role='host')