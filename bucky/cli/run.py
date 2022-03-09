"""`bucky run` CLI."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from ..config import BuckyConfig
from ..model.main import main as model_main
from ..postprocess import main as postprocess_main

app = typer.Typer()


@dataclass
class RunArgs:
    cfg: BuckyConfig
    days: int
    seed: int
    n_mc: int
    # TODO add run name (model needs to support it)


@app.callback(invoke_without_command=True)
def run_args(
    ctx: typer.Context,
    days: int = typer.Option(30, "-d", help="Number of days to project forward"),
    seed: int = typer.Option(42, "-s", help="Global PRNG seed"),
    n_mc: int = typer.Option(100, "-n", help="Number of Monte Carlo iterations"),
):
    ctx.obj = RunArgs(ctx.obj, days, seed, n_mc)

    # If no subcommand is selected run model->postprocess
    if ctx.invoked_subcommand is None:
        raw_output_dir = model(ctx)

        cfg = ctx.obj.cfg
        cfg["postprocessing.run_dir"] = raw_output_dir
        cfg["postprocessing.run_name"] = raw_output_dir.stem
        cfg["postprocessing.output_dir"] = cfg["system.output_dir"] / cfg["postprocessing.run_name"]
        postprocess_main(cfg)


@app.command("model")
def model(ctx: typer.Context):
    """`bucky run model`, run the model itself, dumping raw monte carlo output to raw_output_dir."""
    cfg = ctx.obj.cfg
    cfg["runtime.t_max"] = ctx.obj.days
    cfg["runtime.seed"] = ctx.obj.seed
    cfg["runtime.n_mc"] = ctx.obj.n_mc
    ret = model_main(cfg)
    return ret


@app.command("postprocess")
def postprocess(
    ctx: typer.Context,
    run_dir: Optional[Path] = typer.Option(None, help="Raw output directory for the run being postprocessed"),
):
    """`bucky run postprocess`, aggregate outputs from raw_output_dir to make quantile outputs in output_dir."""
    cfg = ctx.obj.cfg

    # if the raw output dir isn't given, use the most recently created one in the raw_output_dir directory
    if run_dir is None:
        base_dir = cfg["system.raw_output_dir"]
        run_dir = sorted(base_dir.iterdir(), key=lambda path: path.stat().st_ctime)[-1]

    cfg["postprocessing.run_dir"] = run_dir
    cfg["postprocessing.run_name"] = run_dir.stem  # TODO make option
    cfg["postprocessing.output_dir"] = cfg["system.output_dir"] / cfg["postprocessing.run_name"]  # TODO make option
    # add option to change levels (and override the default in the cfg file)

    postprocess_main(cfg)  # TODO only part of cfg? maybe just a few vars?
