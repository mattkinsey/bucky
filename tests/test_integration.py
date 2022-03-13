from pathlib import Path

import pytest
from pytest_steps import test_steps

from bucky.cli.main import main
from bucky.config import locate_base_config


@pytest.mark.integration
@test_steps("bucky data sync", "bucky run model", "bucky run postprocess", "bucky viz plot")
def test_full_run_integration(cli, caplog, tmp_cwd):
    """Perform a simple run of the standard functionality."""

    # Make a tmp conf dir
    conf_dir = Path.cwd() / "bucky.conf.d"
    conf_dir.mkdir()

    # symlink the base_config in
    base_cfg = locate_base_config()
    base_cfg_link = conf_dir / "00_base_cfg"
    base_cfg_link.symlink_to(base_cfg)

    # symlink in any overrides from tests/integration_config
    integration_cfg = base_cfg.parent.parent / "tests" / "integration_config"
    integration_cfg_link = conf_dir / "01_integ_cfg"
    integration_cfg_link.symlink_to(integration_cfg)

    main_flags = [
        "-d",
        "-c",
        str(conf_dir),
    ]

    result = cli.invoke(main, main_flags + ["data", "sync"], echo=True, catch_exceptions=False)
    assert result.exit_code == 0
    yield

    result = cli.invoke(main, main_flags + ["run", "-n2", "-d2", "model"], echo=True, catch_exceptions=False)
    assert result.exit_code == 0
    yield

    result = cli.invoke(main, main_flags + ["run", "postprocess"], echo=True, catch_exceptions=False)
    assert result.exit_code == 0
    yield

    result = cli.invoke(main, main_flags + ["viz", "plot", "-np", "1"], echo=True, catch_exceptions=False)
    assert result.exit_code == 0
    yield
