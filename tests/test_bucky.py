"""Tests for `bucky` module."""
from typing import Generator

import pytest

import bucky


@pytest.fixture
def version() -> Generator[str, None, None]:
    """Sample pytest fixture."""
    yield bucky.__version__


def test_version(version: str) -> None:
    """Sample pytest test function with the pytest fixture as an argument."""
    assert version == "1.0.0"
