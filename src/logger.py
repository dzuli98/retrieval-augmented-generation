import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "rag_system", level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


_default_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger()
    return _default_logger


def set_log_level(level: int) -> None:
    get_logger().setLevel(level)
