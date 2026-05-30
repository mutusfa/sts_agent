"""Tests for shared logging configuration."""

from __future__ import annotations

import logging

from sts_agent.logging_config import configure_logging


def test_configure_logging_sts_agent_levels() -> None:
    configure_logging(0)
    assert logging.getLogger("sts_agent").level == logging.WARNING

    configure_logging(1)
    assert logging.getLogger("sts_agent").level == logging.INFO

    configure_logging(2)
    assert logging.getLogger("sts_agent").level == logging.DEBUG


def test_configure_logging_external_libraries_stay_warning_until_vvv() -> None:
    configure_logging(0)
    assert logging.getLogger().level == logging.WARNING

    configure_logging(2)
    assert logging.getLogger().level == logging.WARNING

    configure_logging(3)
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_external_debug_requires_vvv_not_vv() -> None:
    records: list[logging.LogRecord] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    external = logging.getLogger("urllib3.connectionpool")

    configure_logging(2, root_handlers=[CaptureHandler()])
    external.debug("external debug at -vv")
    assert not records

    records.clear()
    logging.getLogger().handlers.clear()
    configure_logging(3, root_handlers=[CaptureHandler()])
    external.debug("external debug at -vvv")
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
