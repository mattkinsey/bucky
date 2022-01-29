"""`bucky cfg` CLI."""
import typer

app = typer.Typer()

from ..config import cfg


@app.command("print")
def print_():
    """`bucky cfg print`, print curretn cfg to stdout."""
    typer.echo(cfg)
