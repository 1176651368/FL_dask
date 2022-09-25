from communication.tools import ConnectLocal

c = ConnectLocal(role='guest')
for i in range(5):
    c.get(role='arbiter')
    c.push(data='a',role='arbiter')