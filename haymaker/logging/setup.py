import logging
import logging.config
import logging.handlers
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import yaml
from tqdm import tqdm

from haymaker.config import CONFIG
from haymaker.misc import default_path

from .queue_logger import setup_logging_queue, shutdown_logging_queue

LOGGING_PATH = CONFIG.get("logging_path", "logs")

log = logging.getLogger(__name__)

module_directory = Path(__file__).parents[0]

level = 5
logging.addLevelName(5, "DATA")
logging.addLevelName(60, "NOTIFY")

logger = logging.getLogger("haymaker")
logger.setLevel(level)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


class TelegramHandler(logging.handlers.HTTPHandler):
    """Send formatted log records to one Telegram chat."""

    def __init__(
        self,
        host,
        url,
        chat_id,
        secure=True,
        method="POST",
        timeout: float = 5.0,
    ):
        super().__init__(host=host, url=url, method=method, secure=secure)
        self.chat_id = chat_id
        self.timeout = timeout

    def mapLogRecord(self, record):
        return {
            "chat_id": self.chat_id,
            "text": self.format(record),
        }

    def emit(self, record):
        """Send a record to Telegram and report rejected deliveries."""
        try:
            import base64
            import urllib.parse

            host = self.host
            h = self.getConnection(host, self.secure)
            h.timeout = self.timeout
            url = self.url
            data = urllib.parse.urlencode(self.mapLogRecord(record))
            if self.method == "GET":
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}{data}"

            h.putrequest(self.method, url)
            i = host.find(":")
            if i >= 0:
                host = host[:i]
            if self.method == "POST":
                h.putheader("Content-type", "application/x-www-form-urlencoded")
                h.putheader("Content-length", str(len(data)))
            if self.credentials:
                s = ("%s:%s" % self.credentials).encode("utf-8")
                s = "Basic " + base64.b64encode(s).strip().decode("ascii")
                h.putheader("Authorization", s)
            h.endheaders()
            if self.method == "POST":
                h.send(data.encode("utf-8"))

            response = h.getresponse()
            if response.status >= 400:
                body = response.read().decode("utf-8", errors="replace")
                sys.stderr.write(
                    "Telegram log delivery failed: "
                    f"{response.status} {response.reason}: {body}\n"
                )
        except Exception:
            self.handleError(record)


class UTCFormatter(logging.Formatter):
    converter = time.gmtime  # type: ignore[assignment]


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
    """Configure logging and move output handlers onto listener threads."""

    shutdown_logging_queue()
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

    logging.config.dictConfig(config)

    if isinstance(level, str):
        level = logging._nameToLevel.get(level.upper(), 0)
    if level:
        logging.getLogger("haymaker").setLevel(level)
    setup_logging_queue()
