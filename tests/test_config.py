import pytest

from bucky.config import locate_base_config


def test_locate_base_config() -> None:
    base_cfg_dir = locate_base_config()
    assert base_cfg_dir.exists()
    assert base_cfg_dir.is_dir()
    files = {f.name for f in base_cfg_dir.glob("*")}
    assert "00_default.yml" in files
