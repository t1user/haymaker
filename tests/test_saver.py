from datetime import datetime, timezone

import pytest
from helpers import wait_for_condition

from haymaker.saver import AbstractBaseSaver, AsyncSaveManager


def test_AbstractBaseSaver_is_abstract():
    with pytest.raises(TypeError):
        AbstractBaseSaver()  # type: ignore


@pytest.fixture
def saver():
    class Saver(AbstractBaseSaver):
        memo = None

        def save(self, data, *args, **kwargs):
            self.memo = self.name_str(*args, **kwargs)

    return Saver


def test_saver_names(saver: type[AbstractBaseSaver]):
    s = saver("price_df", False)
    s.save({}, "all_data", "NQZ3")
    assert s.memo == "price_df_all_data_NQZ3"  # type: ignore


def test_saver_names_one_args_only(saver: type[AbstractBaseSaver]):
    s = saver("price_df", False)
    s.save({}, "all_data")
    assert s.memo == "price_df_all_data"  # type: ignore


def test_saver_names_include_timestamp(saver: type[AbstractBaseSaver]):
    s = saver("price_df", True)
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H_%M")
    s.save({}, "all_data", "NQZ3")
    assert s.memo == f"price_df_{now}_all_data_NQZ3"  # type: ignore


@pytest.mark.asyncio
async def test_saver():
    class FakeSaver(AbstractBaseSaver):
        output = []

        def save(self, data, *args):
            self.__class__.output.append(data)

    class T:
        save = AsyncSaveManager(FakeSaver("irrelevant", False))

    t = T()
    t1 = T()

    t.save("xxx")
    t1.save("yyy")
    assert await wait_for_condition(lambda: FakeSaver.output == ["xxx", "yyy"])


@pytest.mark.asyncio
async def test_class_with_two_savers():
    class FakeSaver(AbstractBaseSaver):
        output = []

        def save(self, data, *args):
            self.__class__.output.append(data)

    class T:
        # it's not the same saver object, only the same type
        # potentially saving to two different files or collections
        save_one = AsyncSaveManager(FakeSaver("irrelevant", False))
        save_two = AsyncSaveManager(FakeSaver("other_irrelevant", False))

    t = T()

    t.save_one("xxx")
    t.save_two("yyy")
    assert await wait_for_condition(lambda: FakeSaver.output == ["xxx", "yyy"])


@pytest.mark.asyncio
async def test_SaveManager_as_non_descriptor():
    class FakeSaver(AbstractBaseSaver):
        output = []

        def save(self, data, *args):
            self.__class__.output.append(data)

    class T:
        def __init__(self, saver):
            self.save = AsyncSaveManager(saver)

    t = T(FakeSaver("irrelevant", False))

    t.save("xxx")

    assert await wait_for_condition(lambda: FakeSaver.output == ["xxx"])
