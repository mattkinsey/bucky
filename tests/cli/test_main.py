from typer.testing import CliRunner

from bucky.cli.main import main

runner = CliRunner()


def test_app():
    result = runner.invoke(
        main,
        [
            "init",
        ],
    )
    assert result.exit_code == 0
    # assert "Hello Camila" in result.stdout
    # assert "Let's have a coffee in Berlin" in result.stdout
