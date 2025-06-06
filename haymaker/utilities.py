import logging

import numpy as np
from arctic import Arctic  # type: ignore
from ib_insync import IB, ContFuture, Future, MarketOrder  # noqa

from .datastore import AbstractBaseStore

log = logging.getLogger(__name__)


def update_details(
    ib: IB, store: AbstractBaseStore, keys: str | list[str] | None = None
) -> None:
    """
    Pull contract details from ib and update metadata in store.

    Args:
    ib: connected IB instance
    store: datastore instance, for which data will be updated
    keys (Optional): keys in datastore, for which data is to be updated,
                     if not given, update all keys
    """
    if keys is None:
        keys = store.keys()
    elif isinstance(keys, str):
        keys = [keys]

    contracts = {}
    for key in keys:
        try:
            contract = eval(store.read_metadata(key)["repr"])
        except TypeError:
            log.exception(f"Metadata missing for {key}")
            continue
        except NameError as e:
            log.exception(f"Wrong name: {e}")
            print(e)
            continue
        contract.update(includeExpired=True)
        contracts[key] = contract
    ib.qualifyContracts(*contracts.values())
    print(f"contracts qualified: {contracts.values()}")
    details = {}
    for k, v in contracts.copy().items():
        try:
            details[k] = ib.reqContractDetails(v)[0]
        except IndexError:
            print(f"Contract unavailable: {k}")
            del contracts[k]
        if details.get(k):
            print(f"Details for {k} loaded.")
        else:
            print(f"Details for {k} unavailable")
    print("All available details collected")

    # get commission levels
    order = MarketOrder("BUY", 1)
    commissions = {}
    for k, v in contracts.items():
        print(f"Pulling commission for: {v}")
        try:
            commissions[k] = ib.whatIfOrder(v, order).commission
        except AttributeError:
            log.exception(f"Commission unavailable for: {k}")
            commissions[k] = np.nan
        print(f"Commission for {k} saved.")

    for c, d in details.items():
        _d = {"name": d.longName, "min_tick": d.minTick, "commission": commissions[c]}
        store.write_metadata(c, _d)

    print("Data written to store.")


def copy_meta(store):
    """For contracts where meta is no longer available from IB, copy
    meta from existing contracts of the same class"""
    rev = store.review("commission")
    data_df = rev[["name", "tradingClass", "commission", "min_tick"]].dropna()
    data_df = data_df.groupby("tradingClass").last()

    data = data_df.to_dict("index")

    for k in store.keys():
        meta = store.read_metadata(k)

        try:
            store.write_metadata(k, data[meta["tradingClass"]])
        except KeyError:
            continue

    print("Data writtten to store")


def quota_checker(store: Arctic) -> None:
    """
    Given arctic datastore print quotas for all libraries.
    """
    for lib in store.list_libraries():
        quota_ = store.get_quota(lib)
        assert quota_
        quota = quota_ / 1024**3
        size = store.get_library(lib).stats()["totals"]["size"] / 1024**3
        usage = size / quota
        print(f"{lib} - quota: {quota:.2f}GB, size: {size:.2f}GB, usage: {usage:.1%}")


def strjoin(*args: str):
    return "".join(args)
