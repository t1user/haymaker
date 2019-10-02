from ib_insync import IB


class IB_connection:
    def __init__(self):
        self.ib = IB()
        self.connect()

    def connect(self):
        for i in range(1, 20):
            try:
                self.ib.connect('127.0.0.1', 4002, clientId=i)
                break
            except Exception as exc:
                print('connection {} busy... up to the next one'.format(i))
                pass

        self.ib.errorEvent += self.onErrorEvent

    def onErrorEvent(self, *args, **kwargs):
        # 1.0 Network disconnect
        # if error_code in [10182, 1102]:
        print('------------------')
        print('Handling Error')
        print(args, kwargs)
        print('------------------')
