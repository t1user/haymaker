import asyncio
from typing import Callable


async def wait_for_condition(
    condition: Callable[[], bool], timeout: float = 0.5, interval: float = 0.0005
) -> bool:
    """
    Wait until condition() returns True or timeout is reached.  Meant
    to be used when condition depends on an external async function
    and needs to wait to give this function a chance to complete.
    """
    if not callable(condition):
        raise ValueError("Condition must be passed as function!")
    result = False
    start = asyncio.get_running_loop().time()
    try:
        while not condition():
            if asyncio.get_running_loop().time() - start > timeout:
                raise TimeoutError("Condition not met in time")
            await asyncio.sleep(interval)
        result = True
    except TimeoutError:
        result = False
    return result
