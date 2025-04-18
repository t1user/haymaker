import sys
from logging import getLogger
from typing import Callable, Literal, TypeAlias

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
            # log.error(f"PACING VIOLATION: {errorString} {errorCode} {contract}")
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


class StartReconnect(IBHandlers):
    def __init__(self, ib: IB, func: Callable, cleanup: Callable) -> None:
        IBHandlers.__init__(self, ib, func)
        self.host = "localhost"
        self.port = 4002  # this is for paper account
        self.get_clientId()

    def get_clientId(self) -> None:
        """Find an unoccupied clientId for connection."""
        for i in range(1, 20):
            self.id = i
            try:
                log.info(f"Will connect with clientId: {i}")
                self.connect()
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
        log.debug("Disconnected!")
        try:
            if self.cleanup:
                log.debug("Will cleanup.")
                self.cleanup()
            # util.sleep(60)
        except Exception as e:
            log.debug(f"exception caught: {e}")
        # log.debug("will attempt reconnection")
        # self.connect()

    def connect(self) -> None:
        """Establish conection while not using watchdog."""
        log.debug("About to establish connection.")
        failed_attempts = 0
        while not self.ib.isConnected():
            try:
                log.debug("Connecting....")
                # this is a blocking method so will get out of it
                # only after disconnection
                self.ib.connect(self.host, self.port, self.id)
                log.debug("Pausing 60 secs before reconnection attempt.")
                util.sleep(60)
            except ConnectionRefusedError as e:
                log.debug(f"While attepting reconnection: {e}")
                failed_attempts += 1
                util.sleep(30)
                log.debug("post sleep...")
            except Exception as e:
                log.debug(f"Connection error: {e}")
                failed_attempts += 1
                if failed_attempts > 10:
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


class StartWait(StartReconnect):
    """
    After disconnection wait for IB to resolve the issue by itself.
    """

    def __init__(self, ib, func, cleanup):
        self._started = False
        super().__init__(ib, func, cleanup)

    def onDisconnected(self, *args):
        log.debug("Disconnected. Waiting for reconnection...")

    def connect(self) -> None:
        failed_attempts = 0
        while not self.ib.isConnected():
            try:
                log.debug("Connecting....")
                # this is a blocking method so will get out of it
                # only after disconnection
                self.ib.connect(self.host, self.port, self.id)
                log.debug("Pausing 60 secs before reconnection attempt.")
                util.sleep(60)
            except ConnectionRefusedError as e:
                log.debug(f"While attepting reconnection: {e}")
                failed_attempts += 1
                util.sleep(30)
                log.debug("post sleep...")
            except Exception as e:
                log.debug(f"Connection error: {e}")
                failed_attempts += 1
                if failed_attempts > 10:
                    log.error("Reconnection attempt failed afer 10 retries.")

    #     try:
    #         self.ib.connect(self.host, self.port, self.id)
    #     except ConnectionRefusedError as e:
    #         log.debug(f"While attepting reconnection: {e}")
    #         util.sleep(30)
    #         log.debug("post sleep...")
    #     except Exception as e:
    #         log.debug(f"Connection error: {e}")

    # def run(self) -> None:
    #     if not self._started:
    #         super().run()
    #         self._started = True
    #     else:
    #         log.debug("waiting for ib to resume sending data...")


Mode: TypeAlias = Literal["watchdog", "reconnect", "wait"]


class Connection:
    def __init__(
        self,
        ib: IB,
        func: Callable,
        cleanup: Callable | None = None,
        run_mode: Mode = "reconnect",
    ) -> None:
        log.debug(f"Running in {run_mode}")
        if run_mode == "watchdog":
            self.watchdog(ib, func, cleanup)
        elif run_mode == "reconnect":
            self.reconnect(ib, func, cleanup)
        elif run_mode == "wait":
            self.wait(ib, func, cleanup)
        else:
            raise ValueError(f"Unknown mode: {run_mode}")

    def watchdog(self, ib, func, cleanup):
        return StartWatchdog(ib, func, cleanup)

    def reconnect(self, ib, func, cleanup):
        return StartReconnect(ib, func, cleanup)

    def wait(self, ib, func, cleanup):
        return StartWait(ib, func, cleanup)
