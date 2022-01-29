"""`bucky viz` CLI."""
import typer

app = typer.Typer()

# from ..config import cfg


@app.command("plot")
def checkout():
    """`bucky viz plot`, produce matplotlib quantile plots from output files."""
    typer.echo("plot")


# @app.command("map")
# def update():
#    typer.echo("map")
