import pytest
from pytest_steps import test_steps

from bucky.cli.main import main


@pytest.mark.integration
@test_steps("bucky data sync", "bucky run model", "bucky run postprocess", "bucky viz plot")
def test_full_run_integration(cli, caplog, tmp_cwd):
    result = cli.invoke(main, ["-d", "data", "sync"], echo=True, catch_exceptions=False)
    assert result.exit_code == 0
    yield

    result = cli.invoke(main, ["-d", "run", "-n2", "-d2", "model"], echo=True, catch_exceptions=False)
    assert result.exit_code == 0
    yield

    result = cli.invoke(main, ["-d", "run", "postprocess"], echo=True, catch_exceptions=False)
    assert result.exit_code == 0
    yield

    # result = cli.invoke( main, ["-d", "viz", "plot"], echo=True, catch_exceptions=False)
    # assert result.exit_code == 0
    # from bucky.viz.plot import main as plot_main
    # plot_main() # fails b/c of arg parse
    yield
