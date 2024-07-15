import asyncio
import sys
from logging import getLogger
from typing import Callable

from ib_insync import IB, Contract, util
from ib_insync.ibcontroller import IBC, Watchdog

log = getLogger(__name__)


class IBHandlers:
    def __init__(self, ib: IB, func: Callable) -> None:
        self.ib = ib
        self.func = func
        self.ib.connectedEvent += self.onConnected
        self.ib.errorEvent += self.onError
        self.ib.client.apiError += self.onApiError
        self.ib.disconnectedEvent += self.onDisconnected

    def onApiError(self, *args) -> None:
        log.error(f"API error: {args}")

    def onConnected(self, *args) -> None:
        log.info("Connected!")

    def onDisconnected(self, *args) -> None:
        pass

    def onError(
        self, reqId: int, errorCode: int, errorString: str, contract: Contract
    ) -> None:
        try:
            what = contract.localSymbol
        except AttributeError:
            what = str(contract)
        log.warning(f"Error {errorCode} {errorString} for: {what}")
        if "pacing violation" in errorString:
            log.error("PACING VIOLATION. Adjust Pacer Parameters.")
            # sys.exit()
        elif errorCode == 1100:
            log.critical("Connection error: figure out how to handle it!")


class StartWatchdog(IBHandlers):
    def __init__(self, ib: IB, func: Callable) -> None:
        log.debug("Initializing watchdog")
        ibc = IBC(
            twsVersion=1023,
            gateway=True,
            tradingMode="paper",
        )
        watchdog = Watchdog(
            ibc,
            ib,
            port=4002,
            clientId=10,
        )
        log.debug("Attaching handlers...")
        IBHandlers.__init__(self, ib, func)
        watchdog.startedEvent += self.onStarted
        log.debug("Initializing watchdog...")
        watchdog.start()
        log.debug("Watchdog started.")
        ib.run()
        log.debug("ib run.")

    def onStarted(self, *args):
        log.debug(f"Starting: {args}")
        util.run(self.func())

    def onDisconnected(self):
        pass


class StartNoWatchdog(IBHandlers):
    def __init__(self, ib: IB, func: Callable) -> None:
        IBHandlers.__init__(self, ib, func)
        self.host = "localhost"
        self.port = 4002  # this is for paper account
        # self.id = randint(2, 1000)
        self.get_clientId()

    def get_clientId(self) -> None:
        """Find an unoccupied clientId for connection."""
        for i in range(1, 20):
            self.id = i
            try:
                self.connect()
                log.info(f"connected with clientId: {i}")
                return
            except ConnectionRefusedError:
                log.error("TWS or IB Gateway is not running.")
                break
            except Exception as exc:
                message = (
                    f"exception {exc} for connection {i}... "
                    "moving up to the next one"
                )
                log.debug(message)

    def onConnected(self):
        log.info("Connected!")
        self.run()

    def onDisconnected(self, *args) -> None:
        """Initiate re-start when external watchdog manages api connection."""
        log.debug("Disconnected!")
        try:
            util.sleep(60)
        except Exception as e:
            log.debug(f"exception caught: {e}")
        log.debug("will attempt reconnection")
        self.connect()

    def connect(self) -> None:
        """Establish conection while not using watchdog."""
        log.debug("Connecting....")
        counter = 0
        while not self.ib.isConnected():
            try:
                self.ib.connect(self.host, self.port, self.id)
            except ConnectionRefusedError as e:
                log.debug(f"While attepting reconnection: {e}")
                util.sleep(30)
                log.debug("post sleep...")
            except Exception as e:
                log.debug(f"Connection error: {e}")
                if counter > 10:
                    log.error("Reconnection attempt failed afer 10 retries.")

    def run(self) -> None:
        """Fire-up dataloding function."""
        log.debug(f"running {self.func}")
        try:
            util.run(self.func())
        except (KeyboardInterrupt, SystemExit):
            self.ib.disconnect()
            sys.exit()
        except Exception as e:
            log.exception(f"ignoring schedule exception: {e}")
            # raise e


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
