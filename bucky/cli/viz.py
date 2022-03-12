"""`bucky viz` CLI."""
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
):
    """`bucky viz plot`, produce matplotlib quantile plots from output files."""
    cfg = ctx.obj
    if input_dir is None:
        base_dir = cfg["system.output_dir"]
        input_dir = sorted(base_dir.iterdir(), key=lambda path: path.stat().st_ctime)[-1]

    cfg["plot.input_dir"] = input_dir
    cfg["plot.levels"] = [level.name for level in levels]
    cfg["plot.columns"] = columns
    cfg["plot.num_proc"] = num_proc

    plot_main(cfg["plot"])
