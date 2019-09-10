from ib_insync import IB


class IB_connection:
    def __init__(self):
        self.ib = IB()
        self.connect()

    def connect(self):
        self.ib.connect('127.0.0.1', 4002, clientId=1)
