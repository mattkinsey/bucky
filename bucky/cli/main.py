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

import typer

from .cfg import app as cfg_app
from .data import app as data_app
from .run import app as run_app
from .viz import app as viz_app

main = typer.Typer()

main.add_typer(cfg_app, name="cfg")
main.add_typer(data_app, name="data")
main.add_typer(run_app, name="run")
main.add_typer(viz_app, name="viz")


@main.command("init")
def init():
    """."""
    typer.echo(f"init")


if __name__ == "__main__":
    main()
