import logging
import logging.config
import sys
import time
from typing import Optional, Union

import yaml

from . import setup_logging_queue

log = logging.getLogger(__name__)


level = 5
logging.addLevelName(5, "DATA")
logging.addLevelName(60, "NOTIFY")

logger = logging.getLogger("ib_tools")
logger.setLevel(level)


class UTCFormatter(logging.Formatter):
    converter = time.gmtime


def timed_rotating_file_setup(**kwargs):
    filename = kwargs.pop("filename")
    # TODO: This has to be read from config
    dirname = "/home/tomek/ib_data/test_logs"
    full_path = dirname + filename
    return logging.handlers.TimedRotatingFileHandler(filename=full_path, **kwargs)


def setup_logging(
    config_file: Optional[str] = None, level: Union[str, int, None] = None
) -> None:
    if not config_file:
        config_file = "logging_config.yaml"

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f.read())
    except Exception as e:
        logging.basicConfig()
        log.exception(e)
        raise
        sys.exit(1)

    logging.config.dictConfig(config)

    if isinstance(level, str):
        level = logging._nameToLevel[level]
    if level:
        logging.getLogger("ib_tools").setLevel(level)


setup_logging_queue()


# NOT IN USE FOR NOW
class LoggingContext:
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions
