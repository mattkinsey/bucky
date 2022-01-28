#!/usr/bin/env python3
"""Main cli entrypoint for Bucky modules."""
'''
import logging
import runpy
import sys

module_name = sys.argv[1]
del sys.argv[1]

base_pkg_name = "."

if base_pkg_name not in module_name[:6]:
    module_name = base_pkg_name + module_name

try:
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)
except ImportError as e:
    logging.error(e)
    if base_pkg_name not in str(e):
        raise e
    logging.error(" !!! Submodule " + module_name.replace(base_pkg_name, "") + " not found !!!")
    from pkgutil import iter_modules

    from setuptools import find_packages

    def find_modules(path):
        """Recursivly iterate through all submodules."""
        modules = set()
        for pkg in find_packages(path):
            modules.add(pkg)  # noqa: PD005
            pkgpath = path + "/" + pkg.replace(".", "/")
            if sys.version_info.major == 2 or (sys.version_info.major == 3 and sys.version_info.minor < 6):
                for _, name, ispkg in iter_modules([pkgpath]):
                    if not ispkg:
                        modules.add(pkg + "." + name)  # noqa: PD005
            else:
                for info in iter_modules([pkgpath]):
                    if not info.ispkg:
                        modules.add(pkg + "." + info.name)  # noqa: PD005
        return modules

    logging.error("Valid submodules:")
    for mod_name in sorted(find_modules(".")):
        logging.error("    " + mod_name.replace(base_pkg_name, ""))


# TODO add some 'standard' options here like make_input_graph->model->postprocess->plot
'''
from pathlib import Path

import typer

app = typer.Typer()
items_app = typer.Typer()
app.add_typer(items_app, name="items")
users_app = typer.Typer()
app.add_typer(users_app, name="users")


@items_app.command("create")
def items_create(item: str):
    """."""
    typer.echo(f"Creating item: {item}")


@items_app.command("delete")
def items_delete(item: str):
    """."""
    typer.echo(f"Deleting item: {item}")


@items_app.command("sell")
def items_sell(item: str):
    """."""
    typer.echo(f"Selling item: {item}")


@users_app.command("create")
def users_create(user_name: str):
    """."""
    typer.echo(f"Creating user: {user_name}")


@users_app.command("delete")
def users_delete(user_name: str):
    """."""
    typer.echo(f"Deleting user: {user_name}")


main = app

if __name__ == "__main__":
    main()
