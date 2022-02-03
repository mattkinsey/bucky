import contextlib
import os
from pathlib import Path

# TODO see https://stackoverflow.com/a/9213866 for makeing all our contexts into decorators...


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
