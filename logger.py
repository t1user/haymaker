import sys
from datetime import datetime
from logbook import (Logger, StreamHandler, FileHandler, DEBUG, INFO,
                     TimedRotatingFileHandler, set_datetime_format)

from utilities import default_path


log = Logger(__name__)


def logger(name: str, stream_level=DEBUG, file_level=DEBUG,
           folder: str = default_path('logs')) -> Logger:
    set_datetime_format('local')
    StreamHandler(sys.stdout, level=stream_level,
                  bubble=True).push_application()
    filename = __file__.split('/')[-1][:-3]
    FileHandler(
        f'{folder}/{name}_{datetime.today().strftime("%Y-%m-%d_%H-%M")}.log',
        bubble=True, level=file_level, delay=True).push_application()
    return Logger(name)


def rotating_logger_with_shell(name: str, stream_level=DEBUG, file_level=DEBUG,
                               folder: str = default_path('test_logs')
                               ) -> Logger:
    set_datetime_format('local')
    StreamHandler(sys.stdout, level=stream_level,
                  bubble=True).push_application()
    TimedRotatingFileHandler(f'{folder}/{name}.log', date_format='%Y-%m-%d',
                             bubble=True, level=file_level, backup_count=60
                             ).push_application()
    return Logger(name)


def rotating_logger(name: str, level=INFO,
                    folder: str = default_path('test_logs')) -> Logger:
    set_datetime_format('local')
    TimedRotatingFileHandler(f'{folder}/{name}.log', date_format='%Y-%m-%d',
                             bubble=True, level=level, backup_count=60
                             ).push_application()
    return Logger(name)


def log_assert(condition: bool, message: str, module=None):
    """
    Push AssertionError into logger.
    """
    try:
        assert condition is True, message
    except AssertionError:
        log.error(f'{module}: {message}')
        raise
