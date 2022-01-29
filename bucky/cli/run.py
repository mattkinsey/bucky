"""`bucky run` CLI."""
import typer

app = typer.Typer()

# from ..config import cfg


@app.command("model")
def checkout():
    """`bucky run model`, run the model itself, dumping raw monte carlo output to raw_output_dir."""
    typer.echo("model")


@app.command("postprocess")
def update():
    """`bucky run postprocess`, aggregate monte carlo outputs from raw_output_dir to make quantile outputs in output_dir."""
    typer.echo("postprocess")
