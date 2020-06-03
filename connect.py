import asyncio
from typing import Callable

from ib_insync import IB, util, Contract
from ib_insync.ibcontroller import IBC, Watchdog

from logbook import Logger
from handlers import WatchdogHandlers


log = Logger(__name__)


class Connection:
    def __init__(self, ib: IB, func: Callable, watchdog: bool = False) -> None:
        self.watchdog = watchdog
        self.func = func
        self.host = '127.0.0.1'
        self.port = 4002
        self.id = 1
        self.ib = ib
        self.ib.connectedEvent += self.onConnectedEvent
        self.ib.errorEvent += self.onErrorEvent
        self.ib.client.apiError += self.onApiError

        # external watchdog is managing connection
        if not self.watchdog:
            self.ib.disconnectedEvent += self.onDisconnectedEvent
        # watchdog here managing connection
        else:
            self.ib.disconnectedEvent += self.onDisconnectedEventWatchdog

        self.establish_connection()

    def run(self) -> None:
        """Fire-up dataloding function."""
        log.debug(f'running {self.func}')
        try:
            util.run(self.func())
        except Exception as e:
            log.error(f'ignoring schedule exception: {e}')

    def establish_connection(self) -> None:
        """
        Choose appropriate connection path based on whether watchdog
        is to be used.
        """
        log.debug('Establishing connection')
        if self.watchdog:
            log.debug('Watchdog to be run')
            self.run_watchdog()
        else:
            log.debug('No watchdog')
            self.get_clientId()

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

    def run_watchdog(self) -> None:
        log.debug(f'Initializing watchdog')
        ibc = IBC(twsVersion=978,
                  gateway=True,
                  tradingMode='paper',
                  )
        watchdog = Watchdog(ibc, self.ib,
                            port='4002',
                            clientId=self.id,
                            )
        handlers = WatchdogHandlers(watchdog)
        watchdog.start()
        log.debug(f'Watchdog started.')
        self.ib.run()
        log.debug(f'ib run.')

    def onEvent(self, *args):
        log.debug(f'logging event: {args}')

    def onConnectedEvent(self) -> None:
        log.debug(f'Connected!')
        self.run()

    def onDisconnectedEvent(self) -> None:
        """Initiate re-start when external watchdog manages api connection."""
        log.debug(f'Disconnected!')
        try:
            util.sleep(60)
        except Exception as e:
            log.debug(f'exception caught: {e}')
        log.debug(f'will attempt reconnection')
        self.connect()

    def onDisconnectedEventWatchdog(self, *args) -> None:
        log.debug(f'Diconnected! Args: {args}')

    def onErrorEvent(self, reqId: int, errorCode: int, errorString: str,
                     contract: Contract) -> None:
        log.error(f'IB error {errorCode}: {errorString}')
        if errorCode == 1100:
            log.warning(f'Event loop will be stopped!')
            asyncio.get_event_loop().stop()
            self.run()

    def onApiError(self, *args) -> None:
        log.error(f'API error: {args}')
