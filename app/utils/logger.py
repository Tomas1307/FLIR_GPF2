import logging
import sys

from loguru import logger as _loguru_logger


class _InterceptHandler(logging.Handler):
    """Redirect standard library logging to loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1
        _loguru_logger.opt(depth=depth, exception=record.exc_info).bind(
            name=record.name
        ).log(level, record.getMessage())


class _PrintfLogger:
    """Thin wrapper around a bound loguru logger that supports both
    ``%``-style (printf) and ``{}``-style formatting.

    The existing codebase uses ``logger.info("msg %d", val)`` everywhere.
    Loguru only supports ``{}``-style natively, so this wrapper pre-formats
    the message when positional args are given.
    """

    def __init__(self, bound_logger):  # noqa: D107
        self._log = bound_logger

    def _fmt(self, msg, args):
        if args:
            try:
                return msg % args
            except (TypeError, ValueError):
                return msg
        return msg

    def debug(self, msg, *args, **kw):  # noqa: D401
        self._log.opt(depth=1).debug(self._fmt(msg, args), **kw)

    def info(self, msg, *args, **kw):  # noqa: D401
        self._log.opt(depth=1).info(self._fmt(msg, args), **kw)

    def warning(self, msg, *args, **kw):  # noqa: D401
        self._log.opt(depth=1).warning(self._fmt(msg, args), **kw)

    def error(self, msg, *args, **kw):  # noqa: D401
        self._log.opt(depth=1).error(self._fmt(msg, args), **kw)

    def exception(self, msg, *args, **kw):  # noqa: D401
        self._log.opt(depth=1).exception(self._fmt(msg, args), **kw)

    def critical(self, msg, *args, **kw):  # noqa: D401
        self._log.opt(depth=1).critical(self._fmt(msg, args), **kw)


def _setup_once() -> None:
    """Configure loguru sinks and intercept stdlib logging (runs once)."""
    if getattr(_setup_once, "_done", False):
        return
    _loguru_logger.remove()
    _loguru_logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {extra[name]} - {message}",
        level="INFO",
    )
    _loguru_logger.add(
        "pipeline.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {extra[name]} - {message}",
        level="DEBUG",
        rotation="50 MB",
    )
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    _setup_once._done = True  # type: ignore[attr-defined]


def get_logger(name: str) -> _PrintfLogger:
    """Return a logger bound with the given module name.

    Drop-in replacement for the previous ``logging.getLogger(name)`` call.
    Supports both ``%``-style and ``{}``-style formatting so all existing
    call sites work unchanged.

    Args:
        name: Logical name for the logger (typically ``__name__``).

    Returns:
        A ``_PrintfLogger`` instance backed by loguru.
    """
    _setup_once()
    return _PrintfLogger(_loguru_logger.bind(name=name))
