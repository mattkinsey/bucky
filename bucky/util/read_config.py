import pathlib

import yaml

# TODO warn if cwd doesnt have the config...
with (pathlib.Path.cwd() / "config.yml").open(mode="r") as f:
    bucky_cfg = yaml.load(f, yaml.SafeLoader)

bucky_cfg["base_dir"] = str(pathlib.Path.cwd())

# Resolve any relpaths
for k in ("data_dir", "raw_output_dir", "output_dir"):
    path = pathlib.Path(bucky_cfg[k])
    if not path.exists():
        print("Path " + str(path) + " does not exist. Creating...")
        path.mkdir(parents=True)
    bucky_cfg[k] = str(path.resolve())
