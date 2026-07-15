import asyncio
import importlib
import logging
import threading
from pathlib import Path
from unittest.mock import Mock

import eventkit as ev  # type: ignore
import pytest
import yaml

import haymaker.logging as haymaker_logging
from haymaker.logging.handlers import TelegramHandler
from haymaker.logging.setup import (
    setup_asyncio_logging,
    setup_logging_queue,
    shutdown_logging_queue,
)


class _TelegramResponse:
    def __init__(self, status, reason, body):
        self.status = status
        self.reason = reason
        self._body = body

    def read(self):
        return self._body


class _TelegramConnection:
    def __init__(self, response):
        self.response = response
        self.sent_data = b""

    def putrequest(self, method, url):
        self.method = method
        self.url = url

    def putheader(self, header, value):
        pass

    def endheaders(self):
        pass

    def send(self, data):
        self.sent_data = data

    def getresponse(self):
        return self.response


@pytest.mark.parametrize(
    "config_name", ["logging_config.yaml", "dataloader_logging_config.yaml"]
)
def test_builtin_logging_config_components_resolve(config_name):
    """Bundled YAML must resolve every Haymaker logging component."""

    config_path = Path(haymaker_logging.__file__).parent / config_name
    config = yaml.safe_load(config_path.read_text())
    components = [
        section.get("()") or section.get("class")
        for category in ("formatters", "handlers")
        for section in config.get(category, {}).values()
    ]

    for component in components:
        if not component or not component.startswith("haymaker."):
            continue
        module_name, attribute = component.rsplit(".", maxsplit=1)
        assert getattr(importlib.import_module(module_name), attribute)


def test_asyncio_exception_handler_routes_failures_to_haymaker_logging(caplog):
    """Unhandled loop failures should reach configured Haymaker destinations."""

    loop = Mock()
    setup_asyncio_logging(loop)
    handler = loop.set_exception_handler.call_args.args[0]
    exception = RuntimeError("task failed")

    with caplog.at_level(logging.ERROR, logger=setup_asyncio_logging.__module__):
        handler(
            loop,
            {
                "message": "Task exception was never retrieved",
                "exception": exception,
            },
        )

    assert any(
        "Task exception was never retrieved" in record.message
        for record in caplog.records
    )
    assert any(
        record.exc_info and record.exc_info[1] is exception for record in caplog.records
    )


def test_telegram_handler_sends_plain_text_without_parse_mode():
    handler = TelegramHandler(
        host="api.telegram.org",
        url="/bot-token/sendMessage",
        chat_id=123,
    )
    handler.setFormatter(logging.Formatter("%(levelname)s\n\n%(message)s"))
    record = logging.LogRecord(
        "haymaker.aggregators",
        logging.WARNING,
        "/tmp/aggregators.py",
        227,
        "operator: %s",
        ("<built-in function add>",),
        None,
    )

    mapped = handler.mapLogRecord(record)

    assert "parse_mode" not in mapped
    assert "WARNING" in mapped["text"]
    assert "<built-in function add>" in mapped["text"]


def test_telegram_handler_reports_rejected_delivery(capsys, monkeypatch):
    handler = TelegramHandler(
        host="api.telegram.org",
        url="/bot-token/sendMessage",
        chat_id=123,
    )
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    response = _TelegramResponse(
        status=400,
        reason="Bad Request",
        body=b'{"ok":false,"description":"bad html"}',
    )
    connection = _TelegramConnection(response)
    monkeypatch.setattr(handler, "getConnection", lambda host, secure: connection)
    record = logging.LogRecord(
        "haymaker.aggregators",
        logging.WARNING,
        "/tmp/aggregators.py",
        227,
        "operator: %s",
        ("<built-in function add>",),
        None,
    )

    handler.emit(record)

    err = capsys.readouterr().err
    assert "Telegram log delivery failed: 400 Bad Request" in err
    assert "bad html" in err
    assert b"chat_id=123" in connection.sent_data
    assert b"parse_mode" not in connection.sent_data
    assert connection.timeout == handler.timeout


def test_shutdown_logging_queue_stops_listener():
    """Logging output should run on a listener thread and restore handlers."""

    shutdown_logging_queue()
    logger = logging.getLogger("threaded-logging-test")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    emitted = threading.Event()
    emitting_threads = []

    class RecordingHandler(logging.Handler):
        def emit(self, record):
            emitting_threads.append(threading.get_ident())
            emitted.set()

    destination = RecordingHandler()
    logger.handlers = [destination]
    caller_thread = threading.get_ident()

    try:
        setup_logging_queue([logger])
        logger.info("threaded record")
        assert emitted.wait(timeout=1)
    finally:
        shutdown_logging_queue()

    assert len(emitting_threads) == 1
    assert emitting_threads[0] != caller_thread
    assert logger.handlers == [destination]


def test_slow_optional_handler_does_not_block_other_handlers():
    """Each configured destination should have an independent listener."""

    shutdown_logging_queue()
    logger = logging.getLogger("isolated-logging-handlers-test")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    slow_started = threading.Event()
    release_slow = threading.Event()
    fast_emitted = threading.Event()

    class SlowHandler(logging.Handler):
        def emit(self, record):
            slow_started.set()
            release_slow.wait(timeout=1)

    class FastHandler(logging.Handler):
        def emit(self, record):
            fast_emitted.set()

    logger.handlers = [SlowHandler(), FastHandler()]

    try:
        setup_logging_queue([logger])
        logger.warning("record for both handlers")
        assert slow_started.wait(timeout=1)
        assert fast_emitted.wait(timeout=0.2)
    finally:
        release_slow.set()
        shutdown_logging_queue()


def test_handler_failure_does_not_stop_listener_thread():
    """One destination failure should not prevent later record delivery."""

    shutdown_logging_queue()
    logger = logging.getLogger("resilient-logging-handler-test")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    second_record = threading.Event()

    class FailsOnceHandler(logging.Handler):
        calls = 0

        def emit(self, record):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("messenger unavailable")
            second_record.set()

        def handleError(self, record):
            pass

    logger.handlers = [FailsOnceHandler()]

    try:
        setup_logging_queue([logger])
        logger.warning("first")
        logger.warning("second")
        assert second_record.wait(timeout=1)
    finally:
        shutdown_logging_queue()


@pytest.mark.asyncio
async def test_eventkit_logs_async_callback_failure_without_custom_handler(caplog):
    """EventKit should report callback failures through its own logging path."""

    event = ev.Event()

    class MyError(Exception):
        pass

    async def failing_coroutine(*args):
        raise MyError("This is the error.")

    event += failing_coroutine

    event.emit()
    await asyncio.sleep(0.001)
    assert any("This is the error" in record.message for record in caplog.records)
    assert any(record.levelname == "ERROR" for record in caplog.records)
