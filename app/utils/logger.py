import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and return a configured logger instance.

    The logger writes to *stdout* with a standardised format that includes the
    timestamp, logger name, level, and message.  This replaces all emoji-based
    ``print`` calls used in the original scripts.

    Args:
        name: Logical name for the logger (typically ``__name__``).
        level: Logging level (default ``logging.INFO``).

    Returns:
        A ``logging.Logger`` instance ready for use.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
