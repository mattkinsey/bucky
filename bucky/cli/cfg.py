"""`bucky cfg` CLI."""
import typer

from ..config import cfg

app = typer.Typer()


@app.command("print")
def print_():
    """`bucky cfg print`, print curretn cfg to stdout."""
    typer.echo(cfg)
