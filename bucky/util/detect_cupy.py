"""Detect if cupy is installed in the python env."""
import importlib

from loguru import logger


def cupy_available() -> bool:
    """Detect if cupy is installed in the python env."""
    cupy_spec = importlib.util.find_spec("cupy")
    if cupy_spec is None:
        logger.info("Cupy not found.")
        return False
    else:
        logger.info("Detected cupy.")
        return True
