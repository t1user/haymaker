"""Custom logging handlers and handler factories used by Haymaker."""

from __future__ import annotations

import base64
import logging
import logging.handlers
import sys
import urllib.parse
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from haymaker.misc import default_path


class LocalQueueHandler(logging.handlers.QueueHandler):
    """Enqueue an unformatted record copy for a listener thread."""

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        """Copy a record without formatting it on the calling thread."""

        return copy(record)


class SafeQueueListener(logging.handlers.QueueListener):
    """Keep delivering records after one destination handler failure."""

    def handle(self, record: logging.LogRecord) -> None:
        """Dispatch a record and report handler failures without stopping."""

        record = self.prepare(record)
        for handler in self.handlers:
            if self.respect_handler_level and record.levelno < handler.level:
                continue
            try:
                handler.handle(record)
            except Exception:
                handler.handleError(record)


class TqdmLoggingHandler(logging.Handler):
    """Write formatted log records without disrupting a tqdm progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        """Format and write one record through tqdm."""

        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


class TelegramHandler(logging.handlers.HTTPHandler):
    """Send formatted log records to one Telegram chat."""

    def __init__(
        self,
        host: str,
        url: str,
        chat_id: str | int,
        secure: bool = True,
        method: str = "POST",
        timeout: float = 5.0,
    ) -> None:
        """Configure Telegram delivery through its HTTP bot endpoint."""

        super().__init__(host=host, url=url, method=method, secure=secure)
        self.chat_id = chat_id
        self.timeout = timeout

    def mapLogRecord(self, record: logging.LogRecord) -> dict[str, str | int]:
        """Map a log record to Telegram's ``sendMessage`` fields."""

        return {
            "chat_id": self.chat_id,
            "text": self.format(record),
        }

    def emit(self, record: logging.LogRecord) -> None:
        """Send a record to Telegram and report rejected deliveries."""

        try:
            host = self.host
            connection = self.getConnection(host, self.secure)
            connection.timeout = self.timeout
            url = self.url
            data = urllib.parse.urlencode(self.mapLogRecord(record))
            if self.method == "GET":
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}{data}"

            connection.putrequest(self.method, url)
            if self.method == "POST":
                connection.putheader(
                    "Content-type", "application/x-www-form-urlencoded"
                )
                connection.putheader("Content-length", str(len(data)))
            if self.credentials:
                credentials = ("%s:%s" % self.credentials).encode("utf-8")
                authorization = "Basic " + base64.b64encode(credentials).strip().decode(
                    "ascii"
                )
                connection.putheader("Authorization", authorization)
            connection.endheaders()
            if self.method == "POST":
                connection.send(data.encode("utf-8"))

            response = connection.getresponse()
            if response.status >= 400:
                body = response.read().decode("utf-8", errors="replace")
                sys.stderr.write(
                    "Telegram log delivery failed: "
                    f"{response.status} {response.reason}: {body}\n"
                )
        except Exception:
            self.handleError(record)


def filename_from_kwargs(
    *,
    _haymaker_base_directory: str = "ib_data",
    _haymaker_logging_directory: str = "logs",
    **kwargs: Any,
) -> dict[str, Any]:
    """Resolve a configured handler filename against the logging directory."""

    directory = Path(
        default_path(
            _haymaker_logging_directory,
            base_directory=_haymaker_base_directory,
        )
    )
    if "filename" not in kwargs:
        kwargs["filename"] = str(directory / "haymakerLog")
    elif Path(kwargs["filename"]).parent == Path("."):
        kwargs["filename"] = str(directory / kwargs["filename"])
    return kwargs


def timed_rotating_file_handler_setup(
    **kwargs: Any,
) -> logging.handlers.TimedRotatingFileHandler:
    """Build a timed rotating file handler using Haymaker path defaults."""

    return logging.handlers.TimedRotatingFileHandler(**filename_from_kwargs(**kwargs))


def file_handler_setup(**kwargs: Any) -> logging.FileHandler:
    """Build a timestamped file handler using Haymaker path defaults."""

    kwargs = filename_from_kwargs(**kwargs)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    kwargs["filename"] = f"{kwargs['filename']}_{date_str}"
    return logging.FileHandler(**kwargs)
