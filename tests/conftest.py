import functools
import os
import sys

import pytest
import typer
from _pytest.logging import LogCaptureFixture
from loguru import logger


@pytest.fixture
def cli():
    """Yield a click.testing.CliRunner to invoke the CLI."""
    class_ = typer.testing.CliRunner

    def invoke_wrapper(f):
        """Augment CliRunner.invoke to emit its output to stdout.

        This enables pytest to show the output in its logs on test
        failures.

        """

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            echo = kwargs.pop("echo", False)
            result = f(*args, **kwargs)

            if echo is True:
                sys.stdout.write(result.stdout)
                sys.stderr.write(result.stderr)

            return result

        return wrapper

    class_.invoke = invoke_wrapper(class_.invoke)
    cli_runner = class_(mix_stderr=False)

    yield cli_runner


# Have pytest caplog capture loguru output


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)


# Add integration test marker that only runs with --integration cli arg


def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true", default=False, help="run integration tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark integrations tests to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_integ = pytest.mark.skip(reason="need --integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integ)
