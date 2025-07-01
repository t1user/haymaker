from datetime import datetime, timezone

import pytest
from helpers import wait_for_condition

from haymaker.misc import default_path
from haymaker.saver import (
    AbstractBaseFileSaver,
    AbstractBaseSaver,
    AsyncSaveManager,
    SyncSaveManager,
)


def test_AbstractBaseSaver_is_abstract():
    with pytest.raises(TypeError):
        AbstractBaseSaver()  # type: ignore


##############################################
# Test if names built correctly
##############################################


@pytest.fixture
def file_saver():

    class FileSaver(AbstractBaseFileSaver):
        _suffix = "filesuffix"

        def save(self, *args):
            pass

    return FileSaver


def test_saver_names(file_saver: type[AbstractBaseFileSaver]):
    class Saver(AbstractBaseFileSaver):

        def save(self, *args):
            pass

    s = Saver("price_df", use_timestamp=False)
    name = s.name_str("price_df", "all_data", "NQZ3")
    assert name == "price_df_all_data_NQZ3"  # type: ignore


def test_filesaver_file_name(file_saver: type[AbstractBaseFileSaver]):
    s = file_saver("price_df", use_timestamp=False)
    filename = s._file()
    assert filename == f"{default_path()}/price_df.filesuffix"


def test_filesaver_names_include_timestamp(
    file_saver: type[AbstractBaseFileSaver],
) -> None:
    s = file_saver("price_df", use_timestamp=True)
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H_%M")
    filename = s._file("all_data", "NQZ3")
    assert filename == f"{default_path()}/price_df_all_data_NQZ3_{now}.filesuffix"


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


##############################################


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
async def test_class_with_two_savers_separate_collections():

    class FakeSaver(AbstractBaseSaver):
        output = {}

        def save(self, data, *args):
            self.__class__.output[self.name_str(self.name)] = data

    class T:
        # it's not the same saver object, only the same type
        # potentially saving to two different files or collections
        save_one = AsyncSaveManager(FakeSaver("collection_one", use_timestamp=False))
        save_two = AsyncSaveManager(FakeSaver("collection_two", use_timestamp=False))

    t = T()

    t.save_one("xxx")
    t.save_two("yyy")
    assert await wait_for_condition(
        lambda: FakeSaver.output == {"collection_one": "xxx", "collection_two": "yyy"}
    )


@pytest.mark.asyncio
async def test_AsyncSaveManager_as_non_descriptor():
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


@pytest.mark.asyncio
async def test_AsyncSaveManager_as_descriptor():
    class FakeSaver(AbstractBaseSaver):
        output = []

        def save(self, data, *args):
            self.__class__.output.append(data)

    class T:
        save = AsyncSaveManager(FakeSaver("irrelevant_name", False))

    t = T()

    t.save("xxx")

    assert await wait_for_condition(lambda: FakeSaver.output == ["xxx"])


def test_SyncSaveManager_as_non_descriptor():
    class FakeSaver(AbstractBaseSaver):
        output = []

        def save(self, data, *args):
            self.__class__.output.append(data)

    class T:
        def __init__(self, saver):
            self.save = SyncSaveManager(saver)

    t = T(FakeSaver("irrelevant", False))

    t.save("xxx")

    assert FakeSaver.output == ["xxx"]


def test_SyncSaveManager_as_descriptor():
    class FakeSaver(AbstractBaseSaver):
        output = []

        def save(self, data, *args):
            self.__class__.output.append(data)

    class T:
        save = SyncSaveManager(FakeSaver("irrelevant_name", False))

    t = T()

    t.save("xxx")

    assert FakeSaver.output == ["xxx"]
