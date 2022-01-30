"""`bucky viz data` CLI."""

import typer
from loguru import logger

app = typer.Typer()


@app.command("checkout")
def checkout(ctx: typer.Context):
    """`bucky data checkout`, perform initial checkout of datasets into data_dir."""
    logger.info(ctx.obj)
    typer.echo("checkout")


@app.command("update")
def update():
    """`bucky data update`, update input datasets."""
    typer.echo("update")
