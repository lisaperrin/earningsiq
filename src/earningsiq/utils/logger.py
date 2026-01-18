import sys
from pathlib import Path
from loguru import logger
from ..config import settings


def setup_logger(log_file: str = "earningsiq.log"):
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.logs_dir / log_file

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        log_path,
        rotation="10 MB",
        retention="1 week",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    return logger


log = setup_logger()
