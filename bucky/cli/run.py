"""`bucky run` CLI."""
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from ..model.main import main as model_main
from ..postprocess import main as postprocess_main

app = typer.Typer()


@app.command("model")
def model(
    ctx: typer.Context,
    days: int = typer.Option(30, "-d", help="Number of days to project forward"),
    seed: int = typer.Option(42, "-s", help="Global PRNG seed"),
    n_mc: int = typer.Option(100, "-n", help="Number of Monte Carlo iterations"),
):
    """`bucky run model`, run the model itself, dumping raw monte carlo output to raw_output_dir."""
    cfg = ctx.obj
    cfg["runtime.t_max"] = days
    cfg["runtime.seed"] = seed
    cfg["runtime.n_mc"] = n_mc
    model_main(cfg)


@app.command("postprocess")
def postprocess(
    ctx: typer.Context,
    run_dir: Optional[Path] = typer.Option(None, help="Raw output directory for the run being postprocessed"),
):
    """`bucky run postprocess`, aggregate outputs from raw_output_dir to make quantile outputs in output_dir."""
    cfg = ctx.obj

    # if the raw output dir isn't given, use the most recently created one in the raw_output_dir directory
    if run_dir is None:
        base_dir = cfg["system.raw_output_dir"]
        run_dir = sorted(base_dir.iterdir(), key=lambda path: path.stat().st_ctime)[-1]

    cfg["postprocessing.run_dir"] = run_dir
    cfg["postprocessing.run_name"] = run_dir.stem  # TODO make option
    cfg["postprocessing.output_dir"] = cfg["system.output_dir"] / cfg["postprocessing.run_name"]  # TODO make option
    # add option to change levels (and override the default in the cfg file)

    postprocess_main(cfg)  # only part of cfg?
