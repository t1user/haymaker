import time
import asyncio

from ib_insync import IB, util
from ib_insync.ibcontroller import IBC, Watchdog

from logbook import Logger


log = Logger(__name__)


class Connection:
    def __init__(self, ib, func, watchdog=False):
        self.watchdog = watchdog
        self.func = func
        self.host = '127.0.0.1'
        self.port = 4002
        self.id = 1
        self.ib = ib
        # self.establish_connection()
        self.ib.connectedEvent += self.onConnectedEvent
        self.ib.errorEvent += self.onErrorEvent
        self.ib.client.apiError += self.onApiError
        # external watchdog is managing connection
        if not self.watchdog:
            self.ib.disconnectedEvent += self.onDisconnectedEvent
        self.establish_connection()

    def run(self):
        log.debug(f'running {self.func}')
        try:
            asyncio.run(self.func())
        except Exception as e:
            log.error(f'ignoring schedule exception: {e}')

    def establish_connection(self):
        log.debug('Establishing connection')
        if self.watchdog:
            log.debug('Watchdog to be run')
            self.run_watchdog()
        else:
            log.debug('No watchdog')
            self.get_clientId()

    def get_clientId(self):
        for i in range(1, 20):
            self.id = i
            try:
                self.connect()
                log.info(f'connected with clientId: {i}')
                break
            except ConnectionRefusedError:
                log.error(f'TWS or IB Gateway is not running.')
                break
            except Exception as exc:
                message = (f'exception {exc} for connection {i}... '
                           'moving up to the next one')
                log.debug(message)

    def connect(self):
        log.debug('Connecting....')
        while not self.ib.isConnected():
            try:
                self.ib.connect(self.host, self.port, self.id)
            except Exception as e:
                log.debug(f'Connection error: {e}')

    def run_watchdog(self):
        log.debug(f'Initializing watchdog')
        asyncio.get_event_loop().set_debug(True)
        ibc = IBC(twsVersion=978,
                  gateway=True,
                  tradingMode='paper',
                  )
        watchdog = Watchdog(ibc, self.ib,
                            port='4002',
                            clientId=self.id,
                            )
        watchdog.start()
        self.ib.run()

    def onEvent(self, *args):
        log.debug(f'logging event: {args}')

    def onConnectedEvent(self):
        log.debug(f'Connected!')
        self.run()

    def onDisconnectedEvent(self):
        log.debug(f'Disconnected!')
        log.debug(f'asyncio tasks: {asyncio.all_tasks()}')
        for task in asyncio.all_tasks():
            log.debug(f'task: {task}')
            try:
                task.cancel()
            except asyncio.CancelledError:
                log.debug(f'task cancelled')
        time.sleep(60)
        self.connect()

    def onErrorEvent(self, *args):
        log.error(f'IB error: {args}')

    def onApiError(self, *args):
        log.error(f'API error: {args}')
