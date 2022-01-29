"""Creates a CSSE formated case/death history file from the data on an input graph."""
# pylint: disable=unsupported-assignment-operation
import argparse
import pickle  # noqa: S403

import networkx as nx
import pandas as pd

parser = argparse.ArgumentParser(
    description="Create CSSE style case/death history file from the data on a pickled Bucky graph",
)
parser.add_argument("graph", type=str, help="graph to read from")
parser.add_argument("output_file", type=str, help="output file (it will be a csv)")

if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.graph, "rb") as f:
        g = pickle.load(f)  # noqa: S301

    adm2 = nx.get_node_attributes(g, "adm2")
    cases = nx.get_node_attributes(g, "case_hist")
    deaths = nx.get_node_attributes(g, "death_hist")
    end_date = g.graph["start_date"]

    case_df = pd.DataFrame.from_dict(cases).stack()
    case_df.index = case_df.index.rename(["date", "adm2"])
    case_df = case_df.rename("cumulative_reported_cases")

    death_df = pd.DataFrame.from_dict(deaths).stack()
    death_df.index = case_df.index.rename(["date", "adm2"])
    death_df = death_df.rename("cumulative_deaths")

    df = pd.DataFrame({"cumulative_reported_cases": case_df, "cumulative_deaths": death_df}).reset_index()
    df["adm2"] = df.adm2.map(adm2)

    dates = pd.date_range(end=end_date, periods=df.date.nunique(), freq="1d")
    df["date"] = df.date.map(dict(enumerate(dates)))

    df.to_csv(args.output_file, index=False)

    # print(df.groupby('date').sum().diff().tail())
