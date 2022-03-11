"""
Tasks for maintaining the project.

Execute 'invoke --list' for guidance on using Invoke
"""
import platform
import webbrowser
from pathlib import Path

from invoke import call, task
from invoke.context import Context
from invoke.runners import Result

ROOT_DIR = Path(__file__).parent
DOCS_DIR = ROOT_DIR.joinpath("docs")
DOCS_BUILD_DIR = DOCS_DIR.joinpath("_build")
DOCS_AUTOAPI_DIR = DOCS_DIR.joinpath("autoapi")
DOCS_INDEX = DOCS_BUILD_DIR.joinpath("index.html")
COVERAGE_FILE = ROOT_DIR.joinpath(".coverage")
COVERAGE_DIR = ROOT_DIR.joinpath("htmlcov")
COVERAGE_REPORT = COVERAGE_DIR.joinpath("index.html")
SOURCE_DIR = ROOT_DIR.joinpath("bucky")
TEST_DIR = ROOT_DIR.joinpath("tests")
PYTHON_TARGETS = [
    SOURCE_DIR,
    TEST_DIR,
    # ROOT_DIR.joinpath("noxfile.py"),
    Path(__file__),
]
PYTHON_TARGETS_STR = " ".join([str(p) for p in PYTHON_TARGETS])


def _run(c: Context, command: str) -> Result:
    return c.run(command, pty=platform.system() != "Windows")


@task()
def clean_build(c):
    # type: (Context) -> None
    """Clean up files from package building."""
    _run(c, "rm -rf build/")
    _run(c, "rm -rf dist/")
    _run(c, "rm -rf .eggs/")
    _run(c, "find . -name '*.egg-info' -exec rm -rf {} +")
    _run(c, "find . -name '*.egg' -exec rm -f {} +")


@task()
def clean_python(c):
    # type: (Context) -> None
    """Clean up python file artifacts."""
    _run(c, "find . -name '*.pyc' -exec rm -f {} +")
    _run(c, "find . -name '*.pyo' -exec rm -f {} +")
    _run(c, "find . -name '*~' -exec rm -f {} +")
    _run(c, "find . -name '__pycache__' -exec rm -rf {} +")


@task()
def clean_tests(c):
    # type: (Context) -> None
    """Clean up files from testing."""
    _run(c, f"rm -f {COVERAGE_FILE}")
    _run(c, f"rm -rf {COVERAGE_DIR}")
    _run(c, "rm -rf .pytest_cache")


@task()
def clean_docs(c):
    # type: (Context) -> None
    """Clean up files from documentation builds."""
    _run(c, f"rm -rf {DOCS_BUILD_DIR}")
    _run(c, f"rm -rf {DOCS_AUTOAPI_DIR}")


@task(pre=[clean_build, clean_python, clean_tests, clean_docs])
def clean(c):
    # type: (Context) -> None
    """Run all clean sub-tasks."""


@task()
def install_hooks(c):
    # type: (Context) -> None
    """Install pre-commit hooks."""
    _run(c, "poetry run pre-commit install")


@task()
def hooks(c):
    # type: (Context) -> None
    """Run pre-commit hooks."""
    _run(c, "poetry run pre-commit run --all-files")


@task(name="format", help={"check": "Checks if source is formatted without applying changes"})
def format_(c, check=False):
    # type: (Context, bool) -> None
    """Format code."""
    isort_options = ["--check-only", "--diff"] if check else []
    _run(c, f"poetry run isort {' '.join(isort_options)} {PYTHON_TARGETS_STR}")
    black_options = ["--diff", "--check"] if check else ["--quiet"]
    _run(c, f"poetry run black {' '.join(black_options)} {PYTHON_TARGETS_STR}")


@task()
def flake8(c):
    # type: (Context) -> None
    """Run flake8."""
    _run(c, f"poetry run flakehell lint {PYTHON_TARGETS_STR}")


@task()
def pylint(c):
    # type: (Context) -> None
    """Run pylint."""
    _run(c, f"poetry run pylint {PYTHON_TARGETS_STR}")


@task()
def safety(c):
    # type: (Context) -> None
    """Run safety."""
    _run(
        c,
        "poetry export --dev --format=requirements.txt --without-hashes | "
        "poetry run safety check --stdin --full-report",
    )


@task(pre=[flake8, safety, pylint, call(format_, check=True)])
def lint(c):
    # type: (Context) -> None
    """Run all linting."""


@task()
def mypy(c):
    # type: (Context) -> None
    """Run mypy."""
    _run(c, f"poetry run mypy {PYTHON_TARGETS_STR}")


@task()
def tests(c):
    # type: (Context) -> None
    """Run tests."""
    pytest_options = ["--xdoctest", "--cov", "--cov-report=", "--cov-fail-under=0"]
    _run(c, f"poetry run pytest {' '.join(pytest_options)} {TEST_DIR} {SOURCE_DIR}")


@task()
def integration(c):
    # type: (Context) -> None
    """Run integration tests."""
    pytest_options = ["--integration", "-o log_cli=true", "--xdoctest", "--cov", "--cov-report=", "--cov-fail-under=0"]
    _run(c, f"poetry run pytest {' '.join(pytest_options)} {TEST_DIR} {SOURCE_DIR}")


@task(
    help={
        "fmt": "Build a local report: report, html, json, annotate, html, xml.",
        "open_browser": "Open the coverage report in the web browser (requires --fmt html)",
    },
)
def coverage(c, fmt="report", open_browser=False):
    # type: (Context, str, bool) -> None
    """Create coverage report."""
    if any(Path().glob(".coverage.*")):
        _run(c, "poetry run coverage combine")
    _run(c, f"poetry run coverage {fmt} -i")
    if fmt == "html" and open_browser:
        webbrowser.open(COVERAGE_REPORT.as_uri())


@task(
    help={
        "serve": "Build the docs watching for changes",
        "open_browser": "Open the docs in the web browser",
    },
)
def docs(c, serve=False, open_browser=False):
    # type: (Context, bool, bool) -> None
    """Build documentation."""
    build_docs = f"sphinx-build -b html {DOCS_DIR} {DOCS_BUILD_DIR}"
    _run(c, build_docs)
    if open_browser:
        webbrowser.open(DOCS_INDEX.absolute().as_uri())
    # if serve:
    #    _run(c, f"poetry run watchmedo shell-command -p '*.rst;*.md' -c '{build_docs}' -R -D .")


@task(
    help={
        "part": "Part of the version to be bumped.",
        "dry_run": "Don't write any files, just pretend. (default: False)",
    },
)
def version(c, part, dry_run=False):
    # type: (Context, str, bool) -> None
    """Bump version."""
    bump_options = ["--dry-run"] if dry_run else []
    _run(c, f"bump2version {' '.join(bump_options)} {part}")
