"""arg parser for bucky.model

This module handles all the CLI argument parsing for bucky.model and autodetects CuPy

"""
import argparse
import glob
import importlib
import os

from .util.read_config import bucky_cfg

cupy_spec = importlib.util.find_spec("cupy")
cupy_found = cupy_spec is not None

most_recent_graph = max(
    glob.glob(bucky_cfg["data_dir"] + "/input_graphs/*.p"), key=os.path.getctime, default='Most recently created graph in <data_dir>/input_graphs'
)

parser = argparse.ArgumentParser(description="Bucky Model")

parser.add_argument(
    "--graph",
    "-g",
    dest="graph_file",
    default=most_recent_graph,
    type=str,
    help="Pickle file containing the graph to run",
)
parser.add_argument(
    "par_file",
    default=bucky_cfg["base_dir"] + "/par/scenario_5.yml",
    nargs="?",
    type=str,
    help="File containing paramters",
)
parser.add_argument(
    "--n_mc", "-n", default=100, type=int, help="Number of runs to do for Monte Carlo"
)
parser.add_argument(
    "--days", "-d", default=40, type=int, help="Length of the runs in days"
)
parser.add_argument("-v", "--verbose", action="count", default=1)
parser.add_argument(
    "-q", "--quiet", action="store_true", help="Supress all console output"
)
parser.add_argument(
    "-c",
    "--cache",
    action="store_true",
    help="Cache python files/par file/graph pickle for the run",
)
parser.add_argument(
    "-nmc",
    "--no_mc",
    action="store_true",
    help="Just do one run with the mean param values",
)  # TODO rename to --mean or something
parser.add_argument(
    "-gpu",
    "--gpu",
    action="store_true",
    default=cupy_found,
    help="Use cupy instead of numpy",
)
parser.add_argument(
    "-o",
    "--output_dir",
    default=bucky_cfg['raw_output_dir'],
    type=str,
    help="Dir to put the output files",
)
