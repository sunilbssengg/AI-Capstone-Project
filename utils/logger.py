"""Central logger configuration using loguru."""
import sys

from loguru import logger

from core.config import settings

logger.remove()
logger.add(sys.stderr, level=settings.LOG_LEVEL, colorize=True)
logger.add(
    settings.resolve(settings.LOG_FILE),
    level=settings.LOG_LEVEL,
    rotation="5 MB",
    retention=5,
    enqueue=True,
)

__all__ = ["logger"]
