"""Shared logging configuration for CLI entry points."""

from __future__ import annotations

import logging

_STS_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def configure_logging(
    verbosity: int,
    *,
    handlers: list[logging.Handler] | None = None,
    root_handlers: list[logging.Handler] | None = None,
    logger_name: str = "sts_agent",
) -> None:
    """Configure application and third-party log levels from ``-v`` count.

    ``sts_agent`` loggers: ``(none)`` = WARNING, ``-v`` = INFO, ``-vv+`` = DEBUG.
    All other loggers (via the root logger): WARNING until ``-vvv``, then DEBUG.
    """
    sts_level = _STS_LEVELS.get(min(verbosity, 2), logging.DEBUG)
    external_level = logging.DEBUG if verbosity >= 3 else logging.WARNING

    root = logging.getLogger()
    root.setLevel(external_level)

    pkg = logging.getLogger(logger_name)
    pkg.setLevel(sts_level)

    if handlers:
        for handler in handlers:
            if handler not in pkg.handlers:
                pkg.addHandler(handler)

    if root_handlers:
        for handler in root_handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
