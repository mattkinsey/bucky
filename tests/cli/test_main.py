from typer.testing import CliRunner

from bucky.cli.main import main

runner = CliRunner()


def test_app():
    result = runner.invoke(
        main,
        [
            "--help",
        ],
    )
    assert result.exit_code == 0
    assert "Usage: bucky [OPTIONS] COMMAND [ARGS]" in result.stdout
