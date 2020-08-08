from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List

from arctic import Arctic
import pandas as pd

from logger import Logger
from utilities import default_path

log = Logger(__name__)


class AbstractBaseSaver(ABC):
    """
    Api for saving data during trading/simulation.
    """

    def __init__(self, note: str = ''):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H_%M')
        if note:
            self.note = f'_{note}'
        else:
            self.note = note

    def name_str(self, what: str, contract_str: str,) -> str:
        """
        Return string under which the data is to be saved. Apart from data
        passed by save method, timestamp and note associated with particular
        trading session/simulation are being added.
        """
        contract_str = contract_str.replace('   ', '_').replace(' ', '_')
        return f'{what}_{self.timestamp}_{contract_str}{self.note}'

    @abstractmethod
    def save(self, df: pd.DataFrame, what: str, contract_str: str) -> None:
        """
        Overwrite existing data in store with df.

        Args:
        ---------
        df: data to be saved
        what: string indicating what data it is
        contract_str: string indicating what contract data is associated with
        """
        pass


class PickleSaver(AbstractBaseSaver):
    """
    Serializer for pickle format.
    """

    def __init__(self, path: Optional[str] = None, note: str = '') -> None:
        """
        Path determines where the pickle file will be stored. File name
        determined by self.name_str.
        """
        if path is None:
            self.path = default_path('log_data')
        else:
            self.path = path
        super().__init__(note)

    def save(self, df: pd.DataFrame, what: str, contract_str: str) -> None:
        df.to_pickle(f'{self.path}/{self.name_str(what, contract_str)}.pickle')

    def __str__(self):
        return f'PickleSaver({self.path}, {self.note})'


class ArcticSaver(AbstractBaseSaver):
    """
    Serializer for Arctic VersionStore.
    """

    def __init__(self, host: str = 'localhost', library: str = 'test_log',
                 note: str = '') -> None:
        """
        Library given at init, collection determined by self.name_str.
        """
        self.host = host
        self.library = library
        self.db = Arctic(host)
        self.db.initialize_library(library)
        self.store = self.db[library]
        super().__init__(note)

    def save(self, df: pd.DataFrame, what: str, contract_str: str,
             note: str = '') -> None:
        self.store.write(self.name_str(what, contract_str), df)

    def keys(self) -> List[str]:
        return self.store.list_symbols()

    def read(self, key: str) -> pd.DataFrame:
        return self.store.read(key)

    def __str__(self):
        return (f'ArcticSaver(host={self.host}, library={self.library}, '
                f'note={self.note})')
