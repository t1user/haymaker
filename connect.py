from ib_insync import IB
from ib_insync.ibcontroller import IBC, Watchdog

from logbook import Logger


log = Logger(__name__)


class IB_connection:
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 4002
        self.ib = IB()
        self.find_connection()
        self.ib.errorEvent += self.onErrorEvent

    def find_connection(self):
        for i in range(1, 20):
            self.id = i
            try:
                self.connect()
                log.info(f'connected with clientId: {i}')
                break
            except ConnectionRefusedError:
                self.run_watchdog()
                log.info(f'connection run by watchdog with clientId: {i}')
                break
            except Exception as exc:
                message = (f'exception {exc} for connection {i}... '
                           'moving up to the next one')
                log.debug(message)

    def connect(self):
        self.ib.connect(self.host, self.port, self.id)

    def run_watchdog(self):
        ibc = IBC(twsVersion=978,
                  gateway=False,
                  tradingMode='paper',
                  )
        watchdog = Watchdog(ibc, self.ib,
                            port='4002',
                            clientId=self.id,
                            )

        watchdog.start()

    def onErrorEvent(self, *args):
        log.error(f'IB error: {args}')
