"""Application logging setup.

Uses the standard library. Does not log secrets. Third-party loggers stay
at INFO or above unless the app log level is DEBUG.
"""

from __future__ import annotations

import logging

_CONFIGURED = False


def configure_logging(level: str) -> None:
    """Configure the root logger once with a simple stream format."""

    global _CONFIGURED
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        root.addHandler(handler)
        _CONFIGURED = True
    else:
        for handler in root.handlers:
            handler.setLevel(numeric_level)

    if numeric_level > logging.DEBUG:
        logging.getLogger("uvicorn.error").setLevel(logging.INFO)
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
        logging.getLogger("watchfiles").setLevel(logging.WARNING)
