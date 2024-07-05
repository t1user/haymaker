import logging
import logging.config
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import yaml

from haymaker.config import CONFIG
from haymaker.misc import default_path

from . import setup_logging_queue

LOGGING_PATH = CONFIG.get("logging_path", "logs")

log = logging.getLogger(__name__)

module_directory = Path(__file__).parents[0]

level = 5
logging.addLevelName(5, "DATA")
logging.addLevelName(60, "NOTIFY")

logger = logging.getLogger("haymaker")
logger.setLevel(level)


class UTCFormatter(logging.Formatter):
    converter = time.gmtime


def filename_from_kwargs(**kwargs):

    # file not given -> use defaults
    if "filename" not in kwargs:
        kwargs["filename"] = "/".join((default_path(LOGGING_PATH), "haymakerLog"))

    # file given witout directory -> assume default directory
    elif str(Path(kwargs["filename"]).parents[0]) == ".":
        kwargs["filename"] = "/".join((default_path(LOGGING_PATH), kwargs["filename"]))

    return kwargs


def timed_rotating_file_setup(**kwargs):
    kwargs = filename_from_kwargs(**kwargs)
    return logging.handlers.TimedRotatingFileHandler(**kwargs)


def file_setup(**kwargs):
    kwargs = filename_from_kwargs(**kwargs)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    kwargs["filename"] = f"{kwargs['filename']}_{date_str}"
    return logging.FileHandler(**kwargs)


def setup_logging(
    config_file: Union[str, Path, None] = None, level: Union[str, int, None] = None
) -> None:
    if not config_file:
        config_file = module_directory / "logging_config.yaml"
    elif Path(config_file).is_file():
        pass
    elif Path(module_directory, config_file).exists():
        config_file = module_directory / config_file
    else:
        raise ValueError(f"Cannot open logging config file: {config_file}")

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
        level = logging._nameToLevel.get(level.upper(), 0)
    if level:
        logging.getLogger("haymaker").setLevel(level)


setup_logging_queue()
