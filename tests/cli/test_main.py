from typer.testing import CliRunner

from bucky.cli.main import main


def test_app(cli):
    result = cli.invoke(
        main,
        [
            "--help",
        ],
    )
    assert result.exit_code == 0
    assert "Usage: bucky [OPTIONS] COMMAND [ARGS]" in result.stdout
