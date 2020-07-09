from abc import ABC, abstractmethod
from datetime import datetime

from arctic import Arctic
import pandas as pd
from logger import Logger


log = Logger(__name__)


class AbstractBaseSaver(ABC):
    """
    Api for saving data during trading/simulation.
    """

    def __init__(self, note: str = ''):
        self.timestamp = str(datetime.now())
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
        return f'{what}_{self.timestamp}_{contract_str}{self.note}'

    @abstractmethod
    def save(self, df: pd.DataFrame, what: str, contract_str: str) -> None:
        """
        Override existing data in store with df.

        Args:
        ---------
        df: data to be saved
        what: string indicating what data it is
        contract_str: string indicating what contract data is associated with
        """
        pass


class PickleSaver(AbstractBaseSaver):
    """
    Serielizer for pickle format.
    """

    def __init__(self, path: str, note: str = '') -> None:
        """
        Path determines where the pickle file will be stored. File name
        determined by self.name_str.
        """
        self.path = path
        super().__init__(note)

    def save(self, df: pd.DataFrame, what: str, contract_str: str) -> None:
        self.df.to_pickle(f'{self.path}/{self.name_str}.pickle')


class ArcticSaver(AbstractBaseSaver):
    """
    Serielizer for Arctic VersionStore.
    """

    def __init__(self, host: str = 'localhost', library: str = 'test_log',
                 note: str = '') -> None:
        """
        Library given at init, collection determined by self.name_str.
        """
        self.db = Arctic(host)
        self.db.initialize_library(library)
        self.store = self.db[library]
        super().__init__(note)

    def save(self, df: pd.DataFrame, what: str, contract_str: str,
             note: str = '') -> None:
        self.store.write(self.name_str)
