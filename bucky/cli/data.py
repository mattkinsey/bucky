"""`bucky viz data` CLI."""

import typer

app = typer.Typer()

# from ..config import cfg


@app.command("checkout")
def checkout():
    """`bucky data checkout`, perform initial checkout of datasets into data_dir."""
    typer.echo("checkout")


@app.command("update")
def update():
    """`bucky data update`, update input datasets."""
    typer.echo("update")
