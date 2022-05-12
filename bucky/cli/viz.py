"""`bucky viz` CLI."""
import multiprocessing
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer

from ..viz.plot import main as plot_main

app = typer.Typer()


class AdmLevel(str, Enum):
    adm0 = "adm0"
    adm1 = "adm1"
    adm2 = "adm2"


@app.command("plot")
def plot(
    ctx: typer.Context,
    input_dir: Optional[Path] = typer.Option(None, help=""),
    levels: List[AdmLevel] = typer.Option(
        ["adm0", "adm1"],
        "--levels",
        "-l",
        case_sensitive=False,
        help="Adm levels to generate plots of",
    ),
    columns: List[str] = typer.Option(
        ["daily_reported_cases", "daily_deaths"],
        "--columns",
        "-c",
        help="Columns to plot",
    ),
    num_proc: int = typer.Option(
        -1,
        "--num_proc",
        "-np",
        help="Number of parallel procs to use",
    ),
    n_hist: int = typer.Option(28, "--nhist", "-nh", help="Number of historical days to include in plot"),
    hist_window_size: int = typer.Option(
        1,
        "--window",
        "-w",
        help="Window size for rolling mean of plotted historical data points",
    ),
):
    """`bucky viz plot`, produce matplotlib quantile plots from output files."""
    cfg = ctx.obj
    if input_dir is None:
        base_dir = cfg["system.output_dir"]
        input_dir = sorted(base_dir.iterdir(), key=lambda path: path.stat().st_ctime)[-1]

    cfg["plot.input_dir"] = input_dir
    cfg["plot.levels"] = [level.name for level in levels]
    cfg["plot.columns"] = columns
    cfg["plot.n_hist"] = n_hist
    cfg["plot.window_size"] = hist_window_size
    # Number of processes for pool
    if num_proc == -1:
        num_proc = multiprocessing.cpu_count() // 2
    cfg["plot.num_proc"] = num_proc

    plot_main(cfg["plot"])
