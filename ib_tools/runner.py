import asyncio
import logging
import time
from typing import Awaitable, Callable

import ib_insync as ibi

log = logging.getLogger(__name__)


ibi.util.patchAsyncio()


class ReconnectionError(Exception):
    pass


class App:
    def __init__(self, ib: ibi.IB, coro: Callable[[], Awaitable[None]]) -> None:
        self.ib = ib
        self.coro = coro
        log.debug("App instiated.")

    async def _run(self) -> None:
        # clientId != 0 will not see orders from other clients
        log.debug("About to connect.")

        with await self.ib.connectAsync(port=4002, clientId=0, timeout=7) as ib:
            log.info("Connected!")

            def onDisconnected():
                log.error("IB disconnected!")

            def onConnected():
                log.info("IB connected (callback)")

            def onError(
                reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
            ) -> None:
                if errorCode not in (2158, 2104, 2107, 2106):
                    log.debug(f"Error: {errorCode}, {errorString}, {contract}")
                # don't change it to elif you dumb fuck
                if errorCode in (1101, 10182, 2105):
                    # 10182 - failed to request live updates
                    # 1101 - connection restored, data lost
                    # 1102 - connection restored, data maintained
                    # consider adding 2105, 2103 ("data connection is broken")
                    log.debug(f"About to disconnect {ib}")
                    ib.disconnect()
                    raise ReconnectionError

            def onApiError(msg) -> None:
                log.error(msg)
                if msg == "Peer closed connection.":
                    for t in asyncio.Task.all_tasks():
                        t.cancel()
                    self.run()

            ib.connectedEvent += onConnected
            ib.disconnectedEvent += onDisconnected
            ib.errorEvent += onError
            ib.client.apiError += onApiError

            try:
                log.debug(f"Will attempt to run coros: {self.coro}")
                # await asyncio.gather(self.coro, return_exceptions=True)
                await self.coro()
            except asyncio.CancelledError:
                log.exception("CancelledError caught.")
            except Exception:
                log.exception("ERROR ON RUNNING CORO")
                raise

        log.info("Out of connection context.")

    def stop(self) -> None:
        self.ib.disconnect()
        log.error("------------ IB stopped -----------------")

    def run(self) -> None:
        while True:
            log.debug("New while loop run...")
            try:
                log.info("Will try to connect...")
                asyncio.run(self._run(), debug=True)
            except ConnectionError:
                log.info("Connection error. Will re-connect in 5 secs.")
                time.sleep(5)
            except ConnectionRefusedError:
                log.info("IB not connected. Will re-attempt connection in 10 secs.")
                time.sleep(10)
            except ValueError as value_error:
                log.info(f"Value error: {value_error}")
            except asyncio.TimeoutError:
                log.info("Asyncio timeout error ignored.")
            except ReconnectionError:
                log.error("Reconnection error ignored. RESTART HERE?")
                time.sleep(5)
            except Exception:
                log.exception("Error")
                break
            except (KeyboardInterrupt, SystemExit):
                log.error("Caught keyboard interrupt error.")
                self.stop()
                log.info("IB stopped.")
                break
        log.info("Out of the main app loop.")
