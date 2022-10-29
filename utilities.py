from typing import Optional, Union, List
from pathlib import Path
from os import makedirs, path

import numpy as np
from ib_insync import IB, MarketOrder, ContFuture, Future  # noqa
from logbook import Logger  # type: ignore
from arctic import Arctic  # type: ignore

from datastore import AbstractBaseStore


log = Logger(__name__)


def update_details(
    ib: IB, store: AbstractBaseStore, keys: Optional[Union[str, List[str]]] = None
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
            log.error(f"Metadata missing for {key}")
            continue
        except NameError as e:
            log.error(f"Wrong name: {e}")
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
            log.error(f"Commission unavailable for: {k}")
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


def default_path(*dirnames: str) -> str:
    """
    Return path created by joining  ~/ib_data/ and recursively all dirnames
    If the path doesn't exist create it.
    Should also work in Windows but not tested.
    """
    home = Path.home()
    dirnames_str = " / ".join(dirnames)
    if not Path.exists(home / "ib_data" / dirnames_str):
        makedirs(home.joinpath("ib_data", *dirnames))
    return path.join(str(home), "ib_data", *dirnames)


def quota_checker(store: Arctic) -> None:
    """
    Given arctic datastore print quotas for all libraries.
    """
    for lib in store.list_libraries():
        quota = store.get_quota(lib) / 1024**3
        size = store.get_library(lib).stats()["totals"]["size"] / 1024**3
        usage = size / quota
        print(f"{lib} - quota: {quota:.2f}GB, size: {size:.2f}GB, usage: {usage:.1%}")
