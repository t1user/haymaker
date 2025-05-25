import random
import sys
from logging import getLogger
from typing import Callable, Generator, Literal, TypeAlias

from ib_insync import IB, Contract, util
from ib_insync.ibcontroller import IBC, Watchdog

log = getLogger(__name__)


class IBHandlers:
    def __init__(self, ib: IB, func: Callable, cleanup: Callable | None = None) -> None:
        self.ib = ib
        self.func = func
        self.cleanup = cleanup
        self.ib.connectedEvent += self.onConnected
        self.ib.errorEvent += self.onErr
        self.ib.client.apiError += self.onApiError
        self.ib.disconnectedEvent += self.onDisconnected

    def _id(self) -> Generator[int, None, None]:
        while True:
            yield random.randint(60, 90)

    @property
    def clientId(self) -> int:
        return next(self._id())

    def onApiError(self, *args) -> None:
        log.error(f"API error: {args}")

    def onConnected(self, *args) -> None:
        log.info("Connected!")

    def onDisconnected(self, *args) -> None:
        log.debug("Disconnected!")

    def onErr(
        self, reqId: int, errorCode: int, errorString: str, contract: Contract
    ) -> None:
        try:
            what = contract.localSymbol
        except AttributeError:
            what = str(contract)
        if errorCode in (2106, 2107):
            return
        elif "pacing violation" in errorString:
            pass
        else:
            log.warning(f"IB warning: {errorCode} {errorString} for: {what}")


class StartWatchdog(IBHandlers):
    def __init__(self, ib: IB, func: Callable, cleanup: Callable | None) -> None:
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
            clientId=self.clientId,
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


class StartReconnect(IBHandlers):
    def __init__(self, ib: IB, func: Callable, cleanup: Callable | None) -> None:
        IBHandlers.__init__(self, ib, func, cleanup)
        self.host = "localhost"
        self.port = 4002  # this is for paper account
        self.connect()

    def onConnected(self):
        """Fire-up dataloding function."""
        log.info("Connected!")
        log.debug(f"running {self.func}")
        try:
            util.run(self.func())
        except (KeyboardInterrupt, SystemExit):
            self.ib.disconnect()
            sys.exit()
        except Exception as e:
            log.exception(f"ignoring schedule exception: {e}")

    def onDisconnected(self, *args) -> None:
        log.debug("Disconnected!")
        try:
            if self.cleanup:
                log.debug("Will cleanup.")
                self.cleanup()

        except Exception as e:
            log.debug(f"exception caught: {e}")
        util.sleep(60)
        log.debug("will attempt reconnection")
        self.connect()

    def connect(self) -> None:
        """Establish conection while not using watchdog."""
        log.debug("About to establish connection.")
        failed_attempts = 0
        while not self.ib.isConnected():
            try:
                log.debug("Connecting....")
                self.ib.connect(self.host, self.port, self.clientId)
                log.debug("Pausing 60 secs before reconnection attempt.")
                util.sleep(60)
            except Exception as e:
                log.debug(f"Connection error: {e}")
                failed_attempts += 1
                if failed_attempts > 10:
                    log.error("Reconnection attempt failed afer 10 retries.")
                    break
        log.debug("Getting out of connect.")


class StartWait(StartReconnect):
    """
    After disconnection wait for IB to resolve the issue by itself.
    """

    def __init__(self, ib: IB, func: Callable, cleanup: Callable | None):
        self._started = False
        super().__init__(ib, func, cleanup)

    def onConnected(self):
        if not self._started:
            super().onConnected()
        else:
            log.debug(
                "Reconnected. Waiting for IB to resuming sending data "
                "(use different `run_mode` setting if this doesn't work). "
            )

    def onDisconnected(self, *args):
        log.debug("Disconnected!")


Mode: TypeAlias = Literal["watchdog", "reconnect", "wait"]


def connection(
    ib, func: Callable, cleanup: Callable | None = None, run_mode: Mode = "reconnect"
):
    log.debug(f"Running in {run_mode}")
    if run_mode == "watchdog":
        return StartWatchdog(ib, func, cleanup)
    elif run_mode == "reconnect":
        return StartReconnect(ib, func, cleanup)
    elif run_mode == "wait":
        return StartWait(ib, func, cleanup)
    else:
        raise ValueError(f"Unknown mode: {run_mode}")
