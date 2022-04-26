"""`bucky run` CLI."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from ..config import BuckyConfig
from ..model.main import main as model_main
from ..postprocess import main as postprocess_main
from ..util.util import generate_runid
from .typer_utils import forward, invoke

app = typer.Typer(context_settings={"ignore_unknown_options": True})


@app.callback(invoke_without_command=True)
def run_no_subcmd(
    ctx: typer.Context,
    days: int = typer.Option(30, "-d", help="Number of days to project forward"),
    seed: int = typer.Option(42, "-s", help="Global PRNG seed"),
    n_mc: int = typer.Option(100, "-n", help="Number of Monte Carlo iterations"),
    run_id: str = typer.Option(generate_runid(), "--runid", help="UUID name of current run"),
):
    # If no subcommand is selected run model->postprocess->plot
    if ctx.invoked_subcommand is None:

        forward(app, model)
        invoke(app, postprocess)

        from .viz import app as viz_app
        from .viz import plot

        invoke(viz_app, plot)


@app.command("model")
def model(
    ctx: typer.Context,
    days: int = typer.Option(30, "-d", help="Number of days to project forward"),
    seed: int = typer.Option(42, "-s", help="Global PRNG seed"),
    n_mc: int = typer.Option(100, "-n", help="Number of Monte Carlo iterations"),
    run_id: str = typer.Option(generate_runid(), "--runid", help="UUID name of current run"),
):
    """`bucky run model`, run the model itself, dumping raw monte carlo output to raw_output_dir."""
    cfg = ctx.obj
    cfg["runtime.t_max"] = days
    cfg["runtime.seed"] = seed
    cfg["runtime.n_mc"] = n_mc
    cfg["runtime.run_id"] = run_id
    ret = model_main(cfg)
    return ret


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

    postprocess_main(cfg)  # TODO only part of cfg? maybe just a few vars?
