"""Parser for the yaml path config file."""
import logging
import pathlib
import sys

import yaml

if "sphinx" in sys.modules:
    # replace with human readable descriptions if generating docs
    bucky_cfg = {}
    for k in ("base_dir", "data_dir", "raw_output_dir", "output_dir", "cache_dir"):
        bucky_cfg[k] = "config.yml: <" + k + ">"

else:
    # TODO warn if cwd doesnt have the config...
    with (pathlib.Path.cwd() / "config.yml").open(mode="r") as f:
        bucky_cfg = yaml.safe_load(f)

    bucky_cfg["base_dir"] = str(pathlib.Path.cwd())

    # Resolve any relpaths
    for k in ("data_dir", "raw_output_dir", "output_dir", "cache_dir"):
        path = pathlib.Path(bucky_cfg[k])
        if not path.exists():
            logging.info("Path " + str(path) + " does not exist. Creating...")
            path.mkdir(parents=True)
        bucky_cfg[k] = str(path.resolve())
