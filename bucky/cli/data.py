"""`bucky viz data` CLI."""

import typer

from ..util.data_sync import process_datasources

app = typer.Typer()


@app.command("sync")
def sync(ctx: typer.Context):
    """`bucky data checkout`, sync datasources into data_dir."""
    cfg = ctx.obj
    process_datasources(cfg["data_sources"], cfg["system.data_dir"])
    typer.echo("sync complete")  # TODO log
