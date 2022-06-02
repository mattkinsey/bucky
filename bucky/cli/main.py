"""`bucky` CLI main command."""

# Goal:
# bucky init
# bucky cfg (--edit, --print)
# bucky data checkout
# bucky data update
# bucky generate_graph
# bucky run
# bucky run model
# bucky run postprocess
# bucky viz *
# bucky reichcsv type stuff?

import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from ..config import BuckyConfig, locate_current_config
from ..util.detect_cupy import cupy_available
from .cfg import app as cfg_app
from .data import app as data_app
from .run import app as run_app
from .viz import app as viz_app

log_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <magenta>{process.name}:{thread.name}</magenta> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"


main = typer.Typer(name="bucky", add_completion=False)

main.add_typer(cfg_app, name="cfg")
main.add_typer(data_app, name="data")
main.add_typer(run_app, name="run")
main.add_typer(viz_app, name="viz")


@main.callback()
def common(
    ctx: typer.Context,
    cfg_path: Optional[Path] = typer.Option(None, "--cfg", "-c", envvar="BUCKY_CFG", help="Bucky cfg file/dir path."),
    verbose: bool = typer.Option(False, "--verbose", "-v", envvar="BUCKY_VERBOSE", help="Enable INFO level logging."),
    debug: bool = typer.Option(False, "--debug", "-d", envvar="BUCKY_DEBUG", help="Enable DEBUG level logging."),
    color: bool = typer.Option(True, "--no-color", show_default=False, help="Disable colorized output."),
    gpu: bool = typer.Option(
        True,
        "--cpu",
        show_default=False,
        help="Force numpy, otherwise cupy will be used if in it's place (if installed).",
    ),
):
    """Bucky CLI."""

    # Setup logging
    if debug:
        log_level = "DEBUG"
    elif verbose:
        log_level = "INFO"
    else:
        log_level = "WARNING"

    try:
        logger.remove(0)
        logger.add(sys.stderr, colorize=color, level=log_level, format=log_fmt, enqueue=True)
        logger.info("Log level set to {}", log_level)
    except ValueError:
        logger.info("Loguru already initialized")

    # Grab bucky cfg
    if cfg_path is None:
        cfg_path = locate_current_config()
    cfg = BuckyConfig().load_cfg(cfg_path)

    # Need some kind of validation cfg here, make sure paths exists etc

    # add runtime flags to cfg
    cfg["runtime.verbose"] = verbose
    cfg["runtime.debug"] = debug
    cfg["runtime.use_cupy"] = cupy_available() if gpu else False

    # put cfg in typer context for downstream commands
    ctx.obj = cfg
    logger.debug(cfg)


if __name__ == "__main__":
    # with logger.catch(reraise=False, onerror=lambda _: sys.exit(1)):
    main()
