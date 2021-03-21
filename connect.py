import asyncio
from typing import Callable

from ib_insync import IB, util, Contract
from ib_insync.ibcontroller import IBC, Watchdog

from logbook import Logger
from handlers import WatchdogHandlers


log = Logger(__name__)


class IBHandlers:
    def __init__(self, ib: IB) -> None:
        self.ib.connectedEvent += self.onConnected
        self.ib.errorEvent += self.onError
        self.ib.client.apiError += self.onApiError
        self.ib.disconnectedEvent += self.onDisconnected

    def onError(self, reqId: int, errorCode: int, errorString: str,
                contract: Contract) -> None:
        log.error(f'IB error {errorCode}: {errorString}')
        if errorCode == 1100:
            log.warning(f'Event loop will be stopped!')
            asyncio.get_event_loop().stop()
            self.run()

    def onApiError(self, *args) -> None:
        log.error(f'API error: {args}')

    def onConnected(self, *args) -> None:
        pass

    def run(self) -> None:
        """Fire-up dataloding function."""
        log.debug(f'running {self.func}')
        try:
            util.run(self.func())
        except Exception as e:
            log.error(f'ignoring schedule exception: {e}')


class StartWatchdog(IBHandlers):
    def __init__(self, ib: IB, func: Callable) -> None:
        self.func = func
        self.ib = ib
        log.debug('Initializing watchdog')
        ibc = IBC(twsVersion=978,
                  gateway=True,
                  tradingMode='paper',
                  )
        watchdog = Watchdog(ibc, ib,
                            port='4002',
                            clientId=10,
                            )
        log.debug('Attaching handlers...')
        IBHandlers.__init__(self, ib)
        watchdog.startedEvent += self.onStarted
        log.debug('Initializing watchdog...')
        watchdog.start()
        log.debug('Watchdog started.')
        ib.run()
        log.debug('ib run.')

    def onStarted(self, *args):
        log.debug(f'Starting: {args}')
        util.run(self.func())

    def onDisconnected(self):
        pass

    def onError(self, *args):
        log.debug(f'Error: {args}')


class StartNoWatchdog(IBHandlers):
    def __init__(self, ib: IB, func: Callable) -> None:
        IBHandlers.__init__(self, ib)
        pass

    def get_clientId(self) -> None:
        """Find an unoccupied clientId for connection."""
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

    def onDisconnected(self) -> None:
        """Initiate re-start when external watchdog manages api connection."""
        log.debug('Disconnected!')
        try:
            util.sleep(60)
        except Exception as e:
            log.debug(f'exception caught: {e}')
        log.debug('will attempt reconnection')
        self.connect()

    def connect(self) -> None:
        """Establish conection while not using watchdog."""
        log.debug('Connecting....')
        while not self.ib.isConnected():
            try:
                self.ib.connect(self.host, self.port, self.id)
            except ConnectionRefusedError as e:
                log.debug(f'While attepting reconnection: {e}')
                util.sleep(30)
                log.debug(f'post sleep...')
            except Exception as e:
                log.debug(f'Connection error: {e}')


class Connection:
    def __init__(self, ib: IB, func: Callable, watchdog: bool = False) -> None:
        if watchdog:
            self.watchdog(ib, func)
        else:
            self.no_watchdog(ib, func)

    def watchdog(self, ib, func):
        return StartWatchdog(ib, func)

    def no_watchdog(self, ib, func):
        return StartNoWatchdog(ib, func)
