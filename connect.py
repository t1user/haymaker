from ib_insync import IB
from logbook import Logger


log = Logger(__name__)


class IB_connection:
    def __init__(self):
        self.ib = IB()
        self.connect()

    def connect(self):
        for i in range(1, 20):
            try:
                self.ib.connect('127.0.0.1', 4002, clientId=i)
                log.info(f'connected with clientId: {i}')
                break
            except Exception as exc:
                message = (f'exception {exc} for connection {i}... '
                           'moving up to the next one')
                log.debug(message)

        self.ib.errorEvent += self.onErrorEvent

    def onErrorEvent(self, *args):
        log.error(f'IB error: {args}')
